#!/usr/bin/env python3
"""Low-memory selector measurement proxy helpers for static ADAPT artifacts.

These helpers intentionally operate on already available measurement-cache
telemetry.  They do not execute shots, build commutators, or transpile circuits.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from pipelines.scaffold.hh_continuation_generators import (
    build_runtime_split_child_sets,
    build_runtime_split_children,
)
from pipelines.scaffold.hh_continuation_scoring import (
    MeasurementCacheAudit,
    compress_measurement_group_keys,
    measurement_basis_key_covers,
    measurement_group_keys_for_term,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.controller_phase_state import (
    _controller_phase_shots,
)
from pipelines.static_adapt.runtime_split import (
    _phase3_runtime_split_child_set_symmetry_spec,
    _phase3_runtime_split_parent_eligible,
    project_and_deduplicate_runtime_split_child_sets,
)
from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES,
    RouteAChildPaddingConfig,
)

SCHEMA_VERSION = "selector_measurement_cache_stats_v1"
NATIVE_SOURCE = "native_admitted_selector_pre_commit_v1"
LEGACY_HISTORY_SOURCE = "legacy_history_measurement_cache_stats"
CONTROLLER_WORK_SCHEMA_VERSION = "controller_measurement_work_proxy_v1"
CONTROLLER_WORK_EVENT_SCHEMA_VERSION = "controller_measurement_work_event_v1"
CONTROLLER_WORK_SCOPE_VERSION = "static_adapt_phase_event_depth_scope_v1"
CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION = "controller_candidate_work_ledger_v1"
CONTROLLER_CANDIDATE_WORK_LEDGER_COMPLETE_STATUS = "explicit_candidate_work_ledger_v1"
CONTROLLER_CANDIDATE_WORK_LEDGER_MISSING_STATUS = "missing_candidate_work_ledger"
NATIVE_CONTROLLER_WORK_SOURCE = "native_controller_live_decision_work_v1"
LEGACY_CONTROLLER_WORK_FALLBACK_SOURCE = "legacy_admitted_selector_measurement_cache_stats"
OPERATOR_PROBE_EVENT_SCHEMA_VERSION = "paper_i_operator_probe_event_v2"
OPERATOR_PROBE_WORK_CONTRACT_ID = "paper_i_hh_operator_probe_contract_v2"
OPERATOR_PROBE_CHARGE_BASIS = "logical_estimator_request_pre_grouping_v1"
COMMON_EXPOSURE_STAGE = "post_common_eligibility_post_expansion_pre_method_filter"
COMMON_EXPOSURE_POLICY_ID = "trajectory_conditioned_full_child_common_exposure_v1"
DEFAULT_EXPANSION_POLICY_ID = "shortlist_pauli_children_v1"
DEFAULT_ELIGIBILITY_POLICY_ID = "static_adapt_common_eligibility_v1"
DEFAULT_DEDUPLICATION_POLICY_ID = "canonical_pauli_operator_deduplication_v1"
DEFAULT_PROBE_ENUMERATOR_ID = "paper_i_operator_probe_enumerator_v1"
CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION = "controller_measurement_work_numeric_validation_v1"
_PAPER_I_CONTROLLER_PHASES = {"phase0", "phase_0", "phase1", "phase2", "phase3"}


@dataclass(frozen=True)
class ControllerMeasurementWorkRecordRuntime:
    pool: Sequence[Any]
    pool_generator_registry: Mapping[str, Any]
    phase3_enabled: bool
    pool_symmetry_specs: Sequence[Mapping[str, Any] | None]
    problem_key: str
    num_sites: int
    ordering: str
    qpb: int
    phase3_runtime_split_mode_key: str
    phase3_runtime_split_selection_mode_key: str
    phase3_runtime_split_child_set_symmetry_policy_key: str
    phase3_runtime_split_max_subset_size_value: int
    phase3_runtime_split_subset_sizes_value: tuple[int, ...] = (1,)
    fixed_num_particles: tuple[int, int] | None = None
    child_padding_config: RouteAChildPaddingConfig | None = None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _is_present(value: Any) -> bool:
    return value is not None and value != ""


def _controller_numeric_validation(summary: dict[str, Any]) -> dict[str, Any]:
    validation = summary.setdefault(
        "numeric_validation",
        {
            "schema": CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION,
            "status": "ok",
            "missing_required_fields": [],
            "invalid_fields": [],
        },
    )
    if not isinstance(validation, dict):
        validation = {
            "schema": CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION,
            "status": "invalid",
            "missing_required_fields": [],
            "invalid_fields": [
                {
                    "field": "numeric_validation",
                    "reason": "invalid_numeric_validation_payload",
                    "paper_i_blocking": True,
                }
            ],
        }
        summary["numeric_validation"] = validation
    validation.setdefault("schema", CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION)
    validation.setdefault("status", "ok")
    if not isinstance(validation.get("missing_required_fields"), list):
        validation["missing_required_fields"] = []
    if not isinstance(validation.get("invalid_fields"), list):
        validation["invalid_fields"] = []
    return validation


def _event_context(event: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("phase", "event_kind", "work_scope", "source", "source_kind"):
        value = event.get(key)
        if _is_present(value):
            out[key] = str(value)
    return out


def _record_numeric_issue(
    summary: dict[str, Any],
    event: Mapping[str, Any],
    *,
    field: str,
    issue_kind: str,
    reason: str,
    paper_i_blocking: bool,
) -> None:
    validation = _controller_numeric_validation(summary)
    item: dict[str, Any] = {
        **_event_context(event),
        "field": str(field),
        "reason": str(reason),
        "paper_i_blocking": bool(paper_i_blocking),
    }
    if str(field) in event:
        item["value_repr"] = repr(event.get(field))[:160]
    if issue_kind == "missing":
        validation["missing_required_fields"].append(item)
    else:
        validation["invalid_fields"].append(item)
    validation["status"] = "invalid"


def _merge_existing_numeric_validation(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    incoming = event.get("numeric_validation")
    if not isinstance(incoming, Mapping):
        incoming = event.get("controller_numeric_validation")
    if not isinstance(incoming, Mapping):
        return
    validation = _controller_numeric_validation(summary)
    for key in ("missing_required_fields", "invalid_fields"):
        entries = incoming.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            continue
        for entry in entries:
            if isinstance(entry, Mapping):
                validation[key].append(dict(entry))
    if validation["missing_required_fields"] or validation["invalid_fields"]:
        validation["status"] = "invalid"


def _phase_uses_records_with_group_keys(event: Mapping[str, Any]) -> bool:
    # Measurement-group keys are cache/grouping diagnostics. They are not a
    # Paper-I table-facing substitute for typed operator-probe counts.
    return False


def _paper_i_probe_phase_requires_typed_count(event: Mapping[str, Any]) -> bool:
    phase = str(event.get("phase", "") or "")
    if phase not in {"phase1", "phase2", "phase3"}:
        return False
    if any(
        key in event and _is_present(event.get(key))
        for key in ("actual_operator_probe_count", "actual_operator_probe_count_total")
    ):
        return False
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
        if _as_float(event.get(key), 0.0) > 0.0:
            return True
    return False


def _paper_i_blocking_numeric_field(event: Mapping[str, Any], key: str) -> bool:
    phase = str(event.get("phase", "") or "")
    if phase not in _PAPER_I_CONTROLLER_PHASES:
        return False
    if key == "records_with_group_keys":
        return _phase_uses_records_with_group_keys(event)
    return key in {"actual_operator_probe_count", "actual_operator_probe_count_total"}


def _numeric_float_for_sum(
    summary: dict[str, Any],
    event: Mapping[str, Any],
    key: str,
) -> float:
    paper_i_blocking = _paper_i_blocking_numeric_field(event, key)
    if key not in event or not _is_present(event.get(key)):
        if paper_i_blocking:
            _record_numeric_issue(
                summary,
                event,
                field=key,
                issue_kind="missing",
                reason="missing_required_controller_numeric_field",
                paper_i_blocking=True,
            )
        return 0.0
    try:
        parsed = float(event.get(key))
    except Exception:
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="malformed_controller_numeric_field",
            paper_i_blocking=paper_i_blocking,
        )
        return 0.0
    if not math.isfinite(parsed):
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="nonfinite_controller_numeric_field",
            paper_i_blocking=paper_i_blocking,
        )
        return 0.0
    if parsed < 0.0:
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="negative_controller_numeric_field",
            paper_i_blocking=paper_i_blocking,
        )
    return float(parsed)


def _numeric_int_for_sum(
    summary: dict[str, Any],
    event: Mapping[str, Any],
    key: str,
) -> int:
    paper_i_blocking = _paper_i_blocking_numeric_field(event, key)
    if key not in event or not _is_present(event.get(key)):
        return 0
    try:
        parsed = float(event.get(key))
    except Exception:
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="malformed_controller_integer_field",
            paper_i_blocking=paper_i_blocking,
        )
        return 0
    if not math.isfinite(parsed):
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="nonfinite_controller_integer_field",
            paper_i_blocking=paper_i_blocking,
        )
        return 0
    rounded = int(parsed)
    if abs(parsed - float(rounded)) > 1e-9:
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="nonintegral_controller_integer_field",
            paper_i_blocking=paper_i_blocking,
        )
    if parsed < 0.0:
        _record_numeric_issue(
            summary,
            event,
            field=key,
            issue_kind="invalid",
            reason="negative_controller_integer_field",
            paper_i_blocking=paper_i_blocking,
        )
    return _as_int(event.get(key), 0)


def _finalized_numeric_validation(summary: Mapping[str, Any]) -> dict[str, Any]:
    missing = []
    invalid = []
    seen: set[int] = set()

    def collect(node: Mapping[str, Any]) -> None:
        ident = id(node)
        if ident in seen:
            return
        seen.add(ident)
        raw = node.get("numeric_validation")
        if not isinstance(raw, Mapping):
            raw = node.get("controller_numeric_validation")
        if isinstance(raw, Mapping):
            for entry in raw.get("missing_required_fields", []):
                if isinstance(entry, Mapping):
                    missing.append(dict(entry))
            for entry in raw.get("invalid_fields", []):
                if isinstance(entry, Mapping):
                    invalid.append(dict(entry))
        for nested_key in ("by_phase", "per_phase", "by_scope"):
            nested = node.get(nested_key)
            if not isinstance(nested, Mapping):
                continue
            for value in nested.values():
                if isinstance(value, Mapping):
                    collect(value)

    collect(summary)
    status = "invalid" if missing or invalid else "ok"
    paper_blocking_missing = [entry for entry in missing if bool(entry.get("paper_i_blocking", False))]
    paper_blocking_invalid = [entry for entry in invalid if bool(entry.get("paper_i_blocking", False))]
    paper_status = (
        "blocked_invalid_controller_numeric_fields"
        if paper_blocking_missing or paper_blocking_invalid
        else "ok"
    )
    return {
        "schema": CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION,
        "status": status,
        "paper_i_table_work_status": paper_status,
        "missing_required_fields": missing,
        "invalid_fields": invalid,
        "paper_i_blocking_missing_required_fields": paper_blocking_missing,
        "paper_i_blocking_invalid_fields": paper_blocking_invalid,
    }


def _cache_summary(cache: MeasurementCacheAudit) -> Mapping[str, Any]:
    try:
        summary = cache.summary()
    except Exception:
        summary = {}
    return summary if isinstance(summary, Mapping) else {}


def _empty_controller_summary(*, include_events: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": CONTROLLER_WORK_SCHEMA_VERSION,
        "source": NATIVE_CONTROLLER_WORK_SOURCE,
        "source_kind": "native_controller_work",
        "legacy_fallback_used": False,
        "legacy_equivalent_to_controller_work": True,
        "events_count": 0,
        "records_evaluated": 0,
        "records_with_group_keys": 0,
        "groups_total": 0,
        "groups_reused": 0,
        "groups_cache_missed": 0,
        "groups_topup": 0,
        "groups_new": 0,
        "total_groups_new": 0,
        "expanded_measurement_group_probe_count": 0,
        "expanded_measurement_group_probe_count_total": 0,
        "shots_total": 0.0,
        "shots_reused": 0.0,
        "shots_new": 0.0,
        "total_shots_new": 0.0,
        "reuse_count_cost": 0.0,
        "per_phase": {},
        "by_phase": {},
        "by_scope": {},
        "work_scope_count": 0,
        "work_scope_version": CONTROLLER_WORK_SCOPE_VERSION,
        "candidate_work_ledger_schema": CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION,
        "candidate_work_ledger_status": CONTROLLER_CANDIDATE_WORK_LEDGER_MISSING_STATUS,
        "candidate_work_event_count": 0,
        "candidate_work_missing_event_count": 0,
        "candidate_count_total": 0,
        "evaluated_count_total": 0,
        "pre_shortlist_count_total": 0,
        "shortlist_size_total": 0,
        "retained_count_total": 0,
        "rejected_count_total": 0,
        "candidate_work_ledger_scopes": {},
        "numeric_validation": {
            "schema": CONTROLLER_NUMERIC_VALIDATION_SCHEMA_VERSION,
            "status": "ok",
            "missing_required_fields": [],
            "invalid_fields": [],
        },
    }
    if include_events:
        out["events"] = []
    return out


def _scope_token(value: Any) -> str:
    token = str(value)
    return token.replace("|", "/").replace("=", ":").strip() or "na"


def _scope_qualifier_payload(scope_qualifiers: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scope_qualifiers, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in sorted(scope_qualifiers, key=str):
        value = scope_qualifiers.get(key)
        if value is None:
            continue
        key_s = _scope_token(key)
        if isinstance(value, (str, int, float, bool)):
            out[key_s] = value
        else:
            out[key_s] = _scope_token(value)
    return out


def _default_controller_work_scope(
    *,
    phase: str,
    event_kind: str,
    depth: int | None,
    scope_qualifiers: Mapping[str, Any] | None = None,
) -> str:
    parts = [
        "static_adapt",
        f"phase={_scope_token(phase)}",
        f"event={_scope_token(event_kind)}",
        f"depth={_scope_token(depth if depth is not None else 'na')}",
    ]
    for key, value in _scope_qualifier_payload(scope_qualifiers).items():
        parts.append(f"{_scope_token(key)}={_scope_token(value)}")
    return "|".join(parts)


def _parse_controller_work_scope(scope: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(scope or "").split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[str(key)] = str(value)
    return out


def _strict_nonnegative_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    rounded = int(parsed)
    if abs(parsed - rounded) > 1e-9:
        return None
    return int(rounded)


def _batch_union_actual_operator_probe_count(summary: Mapping[str, Any]) -> tuple[int, str]:
    total = 0
    phase = "phase3"
    found_event = False

    events = summary.get("events")
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray)):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if str(event.get("event_kind") or "") != "batch_union_scoring":
                continue
            count = _strict_nonnegative_int(
                event.get("actual_operator_probe_count_total", event.get("actual_operator_probe_count"))
            )
            if count is None:
                continue
            found_event = True
            total += int(count)
            if event.get("phase") not in {None, ""}:
                phase = str(event.get("phase"))

    if found_event:
        return int(total), phase

    by_scope = summary.get("by_scope")
    if isinstance(by_scope, Mapping):
        for scope, entry in by_scope.items():
            if not isinstance(entry, Mapping):
                continue
            fields = _parse_controller_work_scope(scope)
            event_kind = str(entry.get("event_kind") or fields.get("event") or "")
            if event_kind != "batch_union_scoring":
                continue
            count = _strict_nonnegative_int(
                entry.get("actual_operator_probe_count_total", entry.get("actual_operator_probe_count"))
            )
            if count is None:
                continue
            total += int(count)
            entry_phase = entry.get("phase") or fields.get("phase")
            if entry_phase not in {None, ""}:
                phase = str(entry_phase)

    return int(total), phase


def record_joint_selector_workspace_work(
    accumulator: "ControllerMeasurementWorkAccumulator | None",
    *,
    snapshot: Any | None,
    selector_summary: Mapping[str, Any],
    depth: int,
    scope_qualifiers: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Charge newly constructed joint-model matrix elements exactly once."""

    if accumulator is None:
        return None
    workspace = selector_summary.get("geometry_workspace")
    if not isinstance(workspace, Mapping):
        return None
    chargeable = _strict_nonnegative_int(
        workspace.get("query_chargeable_unique_geometry_element_count")
    )
    gradient_repairs = _strict_nonnegative_int(
        workspace.get("query_chargeable_gradient_repair_count")
    )
    if chargeable is None:
        chargeable = 0
    if gradient_repairs is None:
        gradient_repairs = 0
    if chargeable == 0 and gradient_repairs == 0:
        return None
    nominal_shots = int(_controller_phase_shots(snapshot, "phase2", 1))
    if nominal_shots <= 0:
        return None
    search_population_count = int(
        _strict_nonnegative_int(workspace.get("search_population_count")) or 0
    )
    common = {
        "phase": "phase2",
        "group_keys": [],
        "nominal_shots_per_group": nominal_shots,
        "records_with_group_keys": 0,
        "depth": int(depth),
        "shot_phase": "phase2",
        "scope_qualifiers": scope_qualifiers,
        "candidate_count": int(search_population_count),
        "shortlist_size": int(search_population_count),
        "retained_count": int(
            _strict_nonnegative_int(selector_summary.get("selected_cardinality"))
            or 0
        ),
    }
    common_metadata = {
        "selector_schema": str(selector_summary.get("schema") or ""),
        "canonical_selection_stage": str(
            selector_summary.get("canonical_selection_stage") or ""
        ),
        "workspace_fingerprint": str(
            workspace.get("workspace_fingerprint") or ""
        ),
        "workspace_build_mode": str(
            workspace.get("workspace_build_mode") or ""
        ),
        "total_mathematically_required_element_count": int(
            _strict_nonnegative_int(
                workspace.get("total_mathematically_required_element_count")
            )
            or 0
        ),
        "reused_phase2_element_count": int(
            _strict_nonnegative_int(
                workspace.get("reused_phase2_element_count")
            )
            or 0
        ),
        "newly_measured_element_count": int(
            _strict_nonnegative_int(
                workspace.get("newly_measured_element_count")
            )
            or 0
        ),
        "required_element_counts": dict(
            workspace.get("required_element_counts", {})
        ),
        "reused_element_counts": dict(
            workspace.get("reused_element_counts", {})
        ),
        "newly_measured_element_counts": dict(
            workspace.get("newly_measured_element_counts", {})
        ),
        "matrix_cache_hit_element_count": int(
            _strict_nonnegative_int(
                workspace.get("matrix_cache_hit_element_count")
            )
            or 0
        ),
        "matrix_cache_miss_element_count": int(
            _strict_nonnegative_int(
                workspace.get("matrix_cache_miss_element_count")
            )
            or 0
        ),
        "matrix_cache_invalidation_reason_counts": dict(
            workspace.get("matrix_cache_invalidation_reason_counts", {})
        ),
        "required_candidate_pair_count": int(
            _strict_nonnegative_int(
                workspace.get("required_candidate_pair_count")
            )
            or 0
        ),
        "constructed_candidate_pair_count": int(
            _strict_nonnegative_int(
                workspace.get("constructed_candidate_pair_count")
            )
            or 0
        ),
        "reused_cached_candidate_pair_count": int(
            _strict_nonnegative_int(
                workspace.get("reused_cached_candidate_pair_count")
            )
            or 0
        ),
        "joint_pair_cache_hit_count": int(
            _strict_nonnegative_int(
                workspace.get("joint_pair_cache_hit_count")
            )
            or 0
        ),
        "joint_pair_cache_miss_count": int(
            _strict_nonnegative_int(
                workspace.get("joint_pair_cache_miss_count")
            )
            or 0
        ),
        "joint_pair_workers_effective": int(
            _strict_nonnegative_int(
                workspace.get("joint_pair_workers_effective")
            )
            or 0
        ),
    }
    metric_event = None
    if chargeable > 0:
        metric_event = accumulator.record_event(
            **common,
            event_kind="batch_union_scoring",
            records_evaluated=int(chargeable),
            evaluated_count=int(chargeable),
            probe_role="metric",
            actual_operator_probe_count=int(chargeable),
            candidate_work_ledger_scope=(
                "route_a_joint_selector_unique_matrix_elements_v1"
            ),
            event_metadata={
                **common_metadata,
                "query_chargeable_unique_geometry_element_count": int(
                    chargeable
                ),
                "full_unique_geometry_element_count": int(
                    _strict_nonnegative_int(
                        workspace.get(
                            "phase2_joint_geometry_reuse_validation", {}
                        ).get("full_unique_geometry_element_count")
                        if isinstance(
                            workspace.get(
                                "phase2_joint_geometry_reuse_validation"
                            ),
                            Mapping,
                        )
                        else None
                    )
                    or 0
                ),
            },
        )
    gradient_event = None
    if gradient_repairs > 0:
        gradient_event = accumulator.record_event(
            **common,
            event_kind="joint_selector_gradient_repair",
            records_evaluated=int(gradient_repairs),
            evaluated_count=int(gradient_repairs),
            probe_role="gradient",
            actual_operator_probe_count=int(gradient_repairs),
            candidate_work_ledger_scope=(
                "route_a_joint_selector_gradient_repair_v1"
            ),
            event_metadata={
                **common_metadata,
                "query_chargeable_gradient_repair_count": int(
                    gradient_repairs
                ),
            },
        )
    return metric_event or gradient_event


def _phase3_batch_summary_supplement_event(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    batch_summary = row.get("phase3_batch_summary")
    if not isinstance(batch_summary, Mapping):
        return None
    canonical_joint_selector = bool(
        str(batch_summary.get("schema") or "")
        == "route_a_joint_schur_selector_v1"
        or str(batch_summary.get("canonical_selection_stage") or "")
        == "post_child_phase2_joint_selector"
    )
    candidate_batch_eval_count = _strict_nonnegative_int(batch_summary.get("candidate_batch_eval_count"))
    if candidate_batch_eval_count in {None, 0}:
        return None
    chargeable_batch_eval_count = _strict_nonnegative_int(
        (
            batch_summary.get("geometry_workspace", {}).get(
                "query_chargeable_unique_geometry_element_count"
            )
            if isinstance(batch_summary.get("geometry_workspace"), Mapping)
            else None
        )
    )
    if chargeable_batch_eval_count is None:
        chargeable_batch_eval_count = _strict_nonnegative_int(
            batch_summary.get("query_chargeable_batch_subset_count")
        )
    if chargeable_batch_eval_count is None:
        chargeable_batch_eval_count = _strict_nonnegative_int(
            batch_summary.get("candidate_batch_eval_count")
        )
    if chargeable_batch_eval_count is None:
        chargeable_batch_eval_count = int(candidate_batch_eval_count)
    represented_count, phase = _batch_union_actual_operator_probe_count(summary)
    if canonical_joint_selector:
        phase = "phase2"
    supplement_count = int(
        max(0, int(chargeable_batch_eval_count) - int(represented_count))
    )
    if supplement_count <= 0:
        return None
    return {
        "phase": phase or "phase3",
        "event_kind": "batch_union_scoring",
        "events_count": 1,
        "records_evaluated": supplement_count,
        "records_with_group_keys": 0,
        "groups_total": 0,
        "groups_reused": 0,
        "groups_cache_missed": 0,
        "groups_topup": 0,
        "groups_new": 0,
        "total_groups_new": 0,
        "shots_total": 0.0,
        "shots_reused": 0.0,
        "shots_new": 0.0,
        "total_shots_new": 0.0,
        "reuse_count_cost": 0.0,
        "candidate_work_ledger_schema": CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION,
        "candidate_work_ledger_status": CONTROLLER_CANDIDATE_WORK_LEDGER_COMPLETE_STATUS,
        "candidate_work_ledger_scope": (
            "route_a_joint_selector_unique_matrix_elements_v1"
            if canonical_joint_selector
            else "phase3_batch_summary_candidate_batch_eval_count_v1"
        ),
        "candidate_work_event_count": 1,
        "candidate_work_missing_event_count": 0,
        "candidate_count": supplement_count,
        "evaluated_count": supplement_count,
        "operator_probe_event_schema": OPERATOR_PROBE_EVENT_SCHEMA_VERSION,
        "work_contract_id": OPERATOR_PROBE_WORK_CONTRACT_ID,
        "operator_probe_charge_basis": OPERATOR_PROBE_CHARGE_BASIS,
        "probe_role": "metric",
        "actual_operator_probe_count": supplement_count,
        "actual_evaluated_candidate_count": supplement_count,
        "phase3_batch_summary_supplement": {
            "schema": "phase3_batch_summary_measurement_work_supplement_v1",
            "status": "applied",
            "candidate_batch_eval_count": int(candidate_batch_eval_count),
            "query_chargeable_batch_subset_count": int(chargeable_batch_eval_count),
            "query_charge_basis": (
                "unique_full_geometry_elements"
                if isinstance(batch_summary.get("geometry_workspace"), Mapping)
                and batch_summary.get("geometry_workspace", {}).get(
                    "query_chargeable_unique_geometry_element_count"
                )
                is not None
                else "batch_subset_evaluations"
            ),
            "reused_child_phase2_singleton_subset_count": int(
                _strict_nonnegative_int(
                    batch_summary.get("reused_child_phase2_singleton_subset_count")
                )
                or 0
            ),
            "existing_batch_union_actual_operator_probe_count": int(represented_count),
            "supplement_actual_operator_probe_count": int(supplement_count),
            "policy": (
                "count_missing_joint_matrix_elements_as_branch_local_phase2_metric_work"
                if canonical_joint_selector
                else "count_missing_candidate_batch_evaluations_as_branch_local_phase3_metric_work"
            ),
            "canonical_joint_selector": bool(canonical_joint_selector),
            "canonical_selection_stage": batch_summary.get(
                "canonical_selection_stage"
            ),
        },
    }


def _apply_phase3_batch_summary_supplement(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    supplement = _phase3_batch_summary_supplement_event(row, summary)
    if supplement is None:
        return dict(summary)
    out = deepcopy(summary)
    phase = str(supplement.get("phase") or "phase3")
    phase_map = out.setdefault("by_phase", out.get("per_phase", {}))
    if not isinstance(phase_map, dict):
        phase_map = {}
        out["by_phase"] = phase_map
    phase_entry = phase_map.setdefault(phase, _empty_controller_summary(include_events=False))
    if not isinstance(phase_entry, dict):
        phase_entry = _empty_controller_summary(include_events=False)
        phase_map[phase] = phase_entry
    if "actual_operator_probe_count" in phase_entry and "actual_operator_probe_count_total" not in phase_entry:
        phase_entry["actual_operator_probe_count_total"] = phase_entry.get("actual_operator_probe_count")
    if "actual_evaluated_candidate_count" in phase_entry and "actual_evaluated_candidate_count_total" not in phase_entry:
        phase_entry["actual_evaluated_candidate_count_total"] = phase_entry.get("actual_evaluated_candidate_count")
    _merge_controller_event_no_events(phase_entry, supplement)
    out["per_phase"] = phase_map
    out["by_phase"] = phase_map
    out["phase3_batch_summary_measurement_work_supplement"] = supplement[
        "phase3_batch_summary_supplement"
    ]
    return _finalize_controller_summary(out)


def _merge_controller_event(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    _merge_existing_numeric_validation(summary, event)
    phase = str(event.get("phase", "unknown") or "unknown")
    scope = str(event.get("work_scope", "") or "")
    for key in (
        "records_evaluated",
        "records_with_group_keys",
        "groups_total",
        "groups_reused",
        "groups_cache_missed",
        "groups_topup",
        "groups_new",
        "total_groups_new",
        "expanded_measurement_group_probe_count",
        "expanded_measurement_group_probe_count_total",
        "reuse_count_cost",
    ):
        summary[key] = _as_float(summary.get(key), 0.0) + _numeric_float_for_sum(summary, event, key)
    for key in ("shots_total", "shots_reused", "shots_new", "total_shots_new"):
        summary[key] = _as_float(summary.get(key), 0.0) + _numeric_float_for_sum(summary, event, key)
    summary["events_count"] = _as_int(summary.get("events_count"), 0) + _as_int(
        event.get("events_count", 1),
        1,
    )
    _merge_candidate_work_ledger(summary, event)
    _merge_operator_probe_work_ledger(summary, event)
    if isinstance(summary.get("events"), list):
        summary["events"].append(dict(event))

    per_phase = summary.setdefault("per_phase", {})
    if not isinstance(per_phase, dict):
        per_phase = {}
        summary["per_phase"] = per_phase
    phase_summary = per_phase.setdefault(phase, _empty_controller_summary(include_events=False))
    if isinstance(phase_summary, dict):
        _merge_controller_event_no_events(phase_summary, event)
    summary["by_phase"] = summary["per_phase"]

    by_scope = summary.setdefault("by_scope", {})
    if not isinstance(by_scope, dict):
        by_scope = {}
        summary["by_scope"] = by_scope
    if scope:
        scope_summary = by_scope.setdefault(scope, _empty_controller_summary(include_events=False))
        if isinstance(scope_summary, dict):
            _merge_controller_event_no_events(scope_summary, event)
        summary["work_scope_count"] = int(len(by_scope))
    summary.setdefault("work_scope_version", CONTROLLER_WORK_SCOPE_VERSION)


def _candidate_work_scope_counts(event: Mapping[str, Any], *, explicit_events: int) -> dict[str, int]:
    scopes = event.get("candidate_work_ledger_scopes")
    if isinstance(scopes, Mapping):
        out: dict[str, int] = {}
        for key, value in scopes.items():
            count = _as_int(value, 0)
            if count > 0:
                out[str(key)] = int(count)
        if out:
            return out
    scope = str(event.get("candidate_work_ledger_scope", "") or "")
    if scope and int(explicit_events) > 0:
        return {scope: int(explicit_events)}
    return {}


def _merge_candidate_work_ledger(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    event_count = _as_int(event.get("events_count", 1), 1)
    explicit_raw = event.get("candidate_work_event_count")
    missing_raw = event.get("candidate_work_missing_event_count")
    if explicit_raw is None and missing_raw is None:
        complete = (
            str(event.get("candidate_work_ledger_schema", "") or "")
            == CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION
            and str(event.get("candidate_work_ledger_status", "") or "")
            == CONTROLLER_CANDIDATE_WORK_LEDGER_COMPLETE_STATUS
        )
        explicit_events = int(event_count) if complete else 0
        missing_events = 0 if complete else int(event_count)
    else:
        explicit_events = _as_int(explicit_raw, 0)
        missing_events = _as_int(missing_raw, 0)

    summary["candidate_work_event_count"] = _as_int(
        summary.get("candidate_work_event_count"), 0
    ) + int(max(0, explicit_events))
    summary["candidate_work_missing_event_count"] = _as_int(
        summary.get("candidate_work_missing_event_count"), 0
    ) + int(max(0, missing_events))
    for event_key, summary_key in (
        ("candidate_count", "candidate_count_total"),
        ("candidate_count_total", "candidate_count_total"),
        ("evaluated_count", "evaluated_count_total"),
        ("evaluated_count_total", "evaluated_count_total"),
        ("pre_shortlist_count", "pre_shortlist_count_total"),
        ("pre_shortlist_count_total", "pre_shortlist_count_total"),
        ("shortlist_size", "shortlist_size_total"),
        ("shortlist_size_total", "shortlist_size_total"),
        ("retained_count", "retained_count_total"),
        ("retained_count_total", "retained_count_total"),
        ("rejected_count", "rejected_count_total"),
        ("rejected_count_total", "rejected_count_total"),
    ):
        if event_key not in event:
            continue
        summary[summary_key] = _as_int(summary.get(summary_key), 0) + _as_int(event.get(event_key), 0)

    scope_counts = summary.setdefault("candidate_work_ledger_scopes", {})
    if not isinstance(scope_counts, dict):
        scope_counts = {}
        summary["candidate_work_ledger_scopes"] = scope_counts
    for scope, count in _candidate_work_scope_counts(event, explicit_events=int(explicit_events)).items():
        scope_counts[str(scope)] = _as_int(scope_counts.get(str(scope)), 0) + int(count)
    summary.setdefault("candidate_work_ledger_schema", CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION)


def _merge_same_value_field(summary: dict[str, Any], event: Mapping[str, Any], key: str) -> None:
    if key not in event or event.get(key) in {None, ""}:
        return
    incoming = event.get(key)
    if key not in summary or summary.get(key) in {None, ""}:
        summary[key] = incoming
    elif summary.get(key) != incoming:
        summary[key] = "mixed"


def _merge_operator_probe_work_ledger(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    if _paper_i_probe_phase_requires_typed_count(event):
        _record_numeric_issue(
            summary,
            event,
            field="actual_operator_probe_count",
            issue_kind="missing",
            reason="missing_required_typed_operator_probe_count",
            paper_i_blocking=True,
        )
    for event_keys, summary_key in (
        (("actual_operator_probe_count_total", "actual_operator_probe_count"), "actual_operator_probe_count_total"),
        (
            ("common_exposure_operator_probe_count_total", "common_exposure_operator_probe_count"),
            "common_exposure_operator_probe_count_total",
        ),
        (("actual_evaluated_candidate_count_total", "actual_evaluated_candidate_count"), "actual_evaluated_candidate_count_total"),
        (("reused_operator_probe_count_total", "reused_operator_probe_count"), "reused_operator_probe_count_total"),
        (("common_parent_candidate_count_total", "common_parent_candidate_count"), "common_parent_candidate_count_total"),
        (("common_expanded_candidate_count_total", "common_expanded_candidate_count"), "common_expanded_candidate_count_total"),
        (("method_input_candidate_count_total", "method_input_candidate_count"), "method_input_candidate_count_total"),
        (("method_shortlist_candidate_count_total", "method_shortlist_candidate_count"), "method_shortlist_candidate_count_total"),
        (("method_retained_candidate_count_total", "method_retained_candidate_count"), "method_retained_candidate_count_total"),
        (("method_rejected_candidate_count_total", "method_rejected_candidate_count"), "method_rejected_candidate_count_total"),
    ):
        source_key = next((key for key in event_keys if key in event), None)
        if source_key is None:
            continue
        summary[summary_key] = _as_int(summary.get(summary_key), 0) + _numeric_int_for_sum(
            summary,
            event,
            source_key,
        )
    if "actual_operator_probe_count" in event or "actual_operator_probe_count_total" in event:
        summary["actual_operator_probe_count"] = _as_int(summary.get("actual_operator_probe_count_total"), 0)
    if "common_exposure_operator_probe_count" in event or "common_exposure_operator_probe_count_total" in event:
        summary["common_exposure_operator_probe_count"] = _as_int(
            summary.get("common_exposure_operator_probe_count_total"), 0
        )
    for key in (
        "operator_probe_event_schema",
        "work_contract_id",
        "probe_role",
        "operator_probe_charge_basis",
        "common_exposure_stage",
        "common_exposure_policy_id",
        "expansion_policy_id",
        "eligibility_policy_id",
        "deduplication_policy_id",
        "probe_enumerator_id",
        "common_universe_manifest_digest",
        "measurement_reuse_policy",
        "measurement_reuse_validation_status",
    ):
        _merge_same_value_field(summary, event, key)


def _merge_controller_event_no_events(summary: dict[str, Any], event: Mapping[str, Any]) -> None:
    _merge_existing_numeric_validation(summary, event)
    for key in (
        "records_evaluated",
        "records_with_group_keys",
        "groups_total",
        "groups_reused",
        "groups_cache_missed",
        "groups_topup",
        "groups_new",
        "total_groups_new",
        "expanded_measurement_group_probe_count",
        "expanded_measurement_group_probe_count_total",
        "reuse_count_cost",
    ):
        summary[key] = _as_float(summary.get(key), 0.0) + _numeric_float_for_sum(summary, event, key)
    for key in ("shots_total", "shots_reused", "shots_new", "total_shots_new"):
        summary[key] = _as_float(summary.get(key), 0.0) + _numeric_float_for_sum(summary, event, key)
    summary["events_count"] = _as_int(summary.get("events_count"), 0) + _as_int(
        event.get("events_count", 1),
        1,
    )
    _merge_candidate_work_ledger(summary, event)
    _merge_operator_probe_work_ledger(summary, event)


def _merge_nested_controller_summaries(
    target: dict[str, Any],
    nested: Mapping[str, Any] | None,
) -> int:
    if not isinstance(nested, Mapping):
        return 0
    merged = 0
    for key, value in nested.items():
        if not isinstance(value, Mapping):
            continue
        target_summary = target.setdefault(str(key), _empty_controller_summary(include_events=False))
        if not isinstance(target_summary, dict):
            target_summary = _empty_controller_summary(include_events=False)
            target[str(key)] = target_summary
        event_like = dict(value)
        if str(key) in _PAPER_I_CONTROLLER_PHASES:
            event_like.setdefault("phase", str(key))
        _merge_controller_event_no_events(target_summary, event_like)
        merged += 1
    return int(merged)


def _candidate_term_from_record_for_probe_count(
    rec: Mapping[str, Any],
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
) -> AnsatzTerm | None:
    candidate_term = rec.get("candidate_term")
    if isinstance(candidate_term, AnsatzTerm):
        return candidate_term
    idx_raw = rec.get("candidate_pool_index")
    try:
        idx = int(idx_raw)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(runtime.pool):
        term = runtime.pool[idx]
        if isinstance(term, AnsatzTerm):
            return term
    return None


def _controller_work_group_keys_from_records(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
) -> tuple[list[str], int, int]:
    records_list = [dict(rec) for rec in (records or []) if isinstance(rec, Mapping)]
    group_keys: list[str] = []
    records_with_group_keys = 0
    for rec in records_list:
        candidate_term = _candidate_term_from_record_for_probe_count(rec, runtime=runtime)
        if isinstance(candidate_term, AnsatzTerm):
            keys = measurement_group_keys_for_term(candidate_term)
            if keys:
                records_with_group_keys += 1
                group_keys.extend(str(key) for key in keys)
    return group_keys, int(len(records_list)), int(records_with_group_keys)


def _generator_metadata_from_record_for_probe_count(
    rec: Mapping[str, Any],
    candidate_term: AnsatzTerm,
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
) -> Mapping[str, Any] | None:
    feat = rec.get("feature")
    if isinstance(feat, CandidateFeatures) and isinstance(feat.generator_metadata, Mapping):
        return dict(feat.generator_metadata)
    meta = runtime.pool_generator_registry.get(str(candidate_term.label))
    if isinstance(meta, Mapping):
        return dict(meta)
    return None


def _symmetry_spec_from_record_for_probe_count(
    rec: Mapping[str, Any],
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
) -> Mapping[str, Any] | None:
    feat = rec.get("feature")
    if isinstance(feat, CandidateFeatures) and isinstance(feat.symmetry_spec, Mapping):
        return dict(feat.symmetry_spec)
    idx_raw = rec.get("candidate_pool_index")
    try:
        idx = int(idx_raw)
    except (TypeError, ValueError):
        return None
    if (
        runtime.phase3_enabled
        and 0 <= idx < len(runtime.pool_symmetry_specs)
        and isinstance(runtime.pool_symmetry_specs[idx], Mapping)
    ):
        return dict(runtime.pool_symmetry_specs[idx])
    return None


def _logical_operator_probe_count_for_records(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
) -> int:
    return int(
        sum(
            1
            for rec in (records or [])
            if isinstance(rec, Mapping)
            and _candidate_term_from_record_for_probe_count(rec, runtime=runtime) is not None
        )
    )


def _common_exposure_probe_payload_for_records(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
    expand_runtime_split: bool,
) -> dict[str, Any]:
    """Count the common logical probe universe for a controller event.

    This is intentionally an operator-probe ledger, not a measurement-group
    ledger. One parent candidate or admissible child-set candidate is one
    logical estimator request under the Paper-I v2 proxy contract.
    """

    parent_count = 0
    expanded_count = 0
    padding_projection_input_count = 0
    padding_projection_output_count = 0
    padding_projection_zero_count = 0
    padding_projection_deduplicated_count = 0
    digest_rows: list[dict[str, Any]] = []
    for rec in (records or []):
        if not isinstance(rec, Mapping):
            continue
        candidate_term = _candidate_term_from_record_for_probe_count(rec, runtime=runtime)
        if candidate_term is None:
            continue
        parent_count += 1
        expanded_count += 1
        parent_label = str(candidate_term.label)
        generator_meta = _generator_metadata_from_record_for_probe_count(
            rec,
            candidate_term,
            runtime=runtime,
        )
        symmetry_spec_candidate = _symmetry_spec_from_record_for_probe_count(
            rec,
            runtime=runtime,
        )
        runtime_split_child_set_symmetry_spec = _phase3_runtime_split_child_set_symmetry_spec(
            symmetry_spec_candidate,
            policy=str(runtime.phase3_runtime_split_child_set_symmetry_policy_key),
            fallback_preserving=bool(str(runtime.problem_key) == "hh"),
        )
        child_set_labels: list[str] = []
        padding_telemetry: dict[str, Any] | None = None
        if (
            expand_runtime_split
            and _phase3_runtime_split_parent_eligible(
                split_mode=str(runtime.phase3_runtime_split_mode_key),
                selection_mode=str(runtime.phase3_runtime_split_selection_mode_key),
                generator_metadata=generator_meta,
                candidate_term=candidate_term,
            )
        ):
            split_children = build_runtime_split_children(
                parent_label=str(parent_label),
                polynomial=candidate_term.polynomial,
                family_id=str(
                    getattr(rec.get("feature"), "candidate_family", "")
                    if isinstance(rec.get("feature"), CandidateFeatures)
                    else ""
                ),
                num_sites=int(runtime.num_sites),
                ordering=str(runtime.ordering),
                qpb=int(max(1, runtime.qpb)),
                split_mode=str(runtime.phase3_runtime_split_mode_key),
                parent_generator_metadata=generator_meta,
                symmetry_spec=runtime_split_child_set_symmetry_spec,
                fixed_num_particles=runtime.fixed_num_particles,
                hard_guard_required=bool(
                    str(
                        runtime.phase3_runtime_split_child_set_symmetry_policy_key
                    )
                    == "hard_guard"
                ),
            )
            child_sets = build_runtime_split_child_sets(
                parent_label=str(parent_label),
                family_id=str(
                    getattr(rec.get("feature"), "candidate_family", "")
                    if isinstance(rec.get("feature"), CandidateFeatures)
                    else ""
                ),
                num_sites=int(runtime.num_sites),
                ordering=str(runtime.ordering),
                qpb=int(max(1, runtime.qpb)),
                split_mode=str(runtime.phase3_runtime_split_mode_key),
                children=split_children,
                parent_generator_metadata=generator_meta,
                symmetry_spec=runtime_split_child_set_symmetry_spec,
                fixed_num_particles=runtime.fixed_num_particles,
                hard_guard_required=bool(
                    str(
                        runtime.phase3_runtime_split_child_set_symmetry_policy_key
                    )
                    == "hard_guard"
                ),
                subset_sizes=runtime.phase3_runtime_split_subset_sizes_value,
            )
            if (
                runtime.child_padding_config is not None
                and str(runtime.child_padding_config.policy)
                in ROUTE_A_CHILD_PADDING_PROJECTED_GROUPED_POLICIES
            ):
                position_raw = rec.get("position_id")
                feature_raw = rec.get("feature")
                if position_raw is None and isinstance(feature_raw, CandidateFeatures):
                    position_raw = feature_raw.position_id
                child_sets_with_position = [
                    {
                        **dict(child_set),
                        "position_id": (
                            None if position_raw is None else int(position_raw)
                        ),
                    }
                    for child_set in child_sets
                ]
                child_sets, padding_telemetry = (
                    project_and_deduplicate_runtime_split_child_sets(
                        child_sets_with_position,
                        config=runtime.child_padding_config,
                        num_sites=int(runtime.num_sites),
                        ordering=str(runtime.ordering),
                        qpb=int(max(1, runtime.qpb)),
                        fixed_num_particles=runtime.fixed_num_particles,
                    )
                )
                padding_projection_input_count += int(
                    padding_telemetry.get("projection_input_count", 0)
                )
                padding_projection_output_count += int(
                    padding_telemetry.get("retained_candidate_count", 0)
                )
                padding_projection_zero_count += int(
                    padding_telemetry.get("projection_zero_rejection_count", 0)
                )
                padding_projection_deduplicated_count += int(
                    padding_telemetry.get("deduplicated_candidate_count", 0)
                )
            child_set_labels = [str(row.get("candidate_label")) for row in child_sets]
            expanded_count += int(len(child_sets))
        digest_rows.append(
            {
                "parent_label": parent_label,
                "expanded_child_set_labels": child_set_labels,
                "child_padding_projection": (
                    None
                    if padding_telemetry is None
                    else {
                        "policy_effective": padding_telemetry.get(
                            "policy_effective"
                        ),
                        "projection_input_count": int(
                            padding_telemetry.get("projection_input_count", 0)
                        ),
                        "retained_candidate_count": int(
                            padding_telemetry.get("retained_candidate_count", 0)
                        ),
                        "projection_zero_rejection_count": int(
                            padding_telemetry.get(
                                "projection_zero_rejection_count", 0
                            )
                        ),
                        "deduplicated_candidate_count": int(
                            padding_telemetry.get(
                                "deduplicated_candidate_count", 0
                            )
                        ),
                    }
                ),
            }
        )
    digest_payload = {
        "policy": "trajectory_conditioned_full_child_common_exposure_v1",
        "expansion_policy": (
            "shortlist_pauli_children_v1"
            if bool(expand_runtime_split)
            else "identity_no_child_expansion_v1"
        ),
        "parent_count": int(parent_count),
        "expanded_count": int(expanded_count),
        "rows": digest_rows,
        "child_padding_policy": (
            None
            if runtime.child_padding_config is None
            else str(runtime.child_padding_config.policy)
        ),
        "child_padding_projection_input_count": int(
            padding_projection_input_count
        ),
        "child_padding_projection_output_count": int(
            padding_projection_output_count
        ),
        "child_padding_projection_zero_count": int(
            padding_projection_zero_count
        ),
        "child_padding_projection_deduplicated_count": int(
            padding_projection_deduplicated_count
        ),
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "common_parent_candidate_count": int(parent_count),
        "common_expanded_candidate_count": int(expanded_count),
        "common_exposure_operator_probe_count": int(expanded_count),
        "common_universe_manifest_digest": digest,
        "child_padding_policy": digest_payload["child_padding_policy"],
        "child_padding_projection_input_count": int(
            padding_projection_input_count
        ),
        "child_padding_projection_output_count": int(
            padding_projection_output_count
        ),
        "child_padding_projection_zero_count": int(
            padding_projection_zero_count
        ),
        "child_padding_projection_deduplicated_count": int(
            padding_projection_deduplicated_count
        ),
    }


def _record_controller_work_for_records(
    accumulator: "ControllerMeasurementWorkAccumulator | None",
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
    snapshot: Any | None,
    phase: str,
    event_kind: str,
    records: Sequence[Mapping[str, Any]] | None,
    depth_value: int | None = None,
    shot_phase: str | None = None,
    work_scope: str | None = None,
    scope_qualifiers: Mapping[str, Any] | None = None,
    candidate_count: int | None = None,
    evaluated_count: int | None = None,
    pre_shortlist_count: int | None = None,
    shortlist_size: int | None = None,
    retained_count: int | None = None,
    rejected_count: int | None = None,
    probe_role: str | None = None,
    actual_operator_probe_count: int | None = None,
    common_parent_candidate_count: int | None = None,
    common_expanded_candidate_count: int | None = None,
    common_exposure_operator_probe_count: int | None = None,
    common_universe_manifest_digest: str | None = None,
    child_padding_policy: str | None = None,
    child_padding_projection_input_count: int = 0,
    child_padding_projection_output_count: int = 0,
    child_padding_projection_zero_count: int = 0,
    child_padding_projection_deduplicated_count: int = 0,
    common_exposure_stage: str = COMMON_EXPOSURE_STAGE,
    common_exposure_policy_id: str = COMMON_EXPOSURE_POLICY_ID,
    expansion_policy_id: str = DEFAULT_EXPANSION_POLICY_ID,
    eligibility_policy_id: str = DEFAULT_ELIGIBILITY_POLICY_ID,
    deduplication_policy_id: str = DEFAULT_DEDUPLICATION_POLICY_ID,
    probe_enumerator_id: str = DEFAULT_PROBE_ENUMERATOR_ID,
    event_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if accumulator is None:
        return None
    phase_name = str(phase)
    shot_phase_name = str(shot_phase or phase_name)
    nominal_shots = (
        _controller_phase_shots(snapshot, shot_phase_name, 1)
        if shot_phase_name in {"phase1", "phase2", "phase3"}
        else 1
    )
    if int(nominal_shots) <= 0:
        return None
    group_keys, records_evaluated, records_with_group_keys = _controller_work_group_keys_from_records(
        records,
        runtime=runtime,
    )
    if records_evaluated <= 0:
        return None
    event_metadata_out = dict(event_metadata or {})
    event_metadata_out["common_exposure_child_padding"] = {
        "policy": (
            None if child_padding_policy in {None, ""} else str(child_padding_policy)
        ),
        "projection_input_count": int(child_padding_projection_input_count),
        "projection_output_count": int(child_padding_projection_output_count),
        "projection_zero_count": int(child_padding_projection_zero_count),
        "projection_deduplicated_count": int(
            child_padding_projection_deduplicated_count
        ),
    }
    return accumulator.record_event(
        phase=phase_name,
        event_kind=str(event_kind),
        group_keys=group_keys,
        nominal_shots_per_group=int(nominal_shots),
        records_evaluated=int(records_evaluated),
        records_with_group_keys=int(records_with_group_keys),
        depth=None if depth_value is None else int(depth_value),
        shot_phase=shot_phase_name,
        work_scope=work_scope,
        scope_qualifiers=scope_qualifiers,
        candidate_count=candidate_count,
        evaluated_count=evaluated_count,
        pre_shortlist_count=pre_shortlist_count,
        shortlist_size=shortlist_size,
        retained_count=retained_count,
        rejected_count=rejected_count,
        probe_role=probe_role,
        actual_operator_probe_count=actual_operator_probe_count,
        common_parent_candidate_count=common_parent_candidate_count,
        common_expanded_candidate_count=common_expanded_candidate_count,
        common_exposure_operator_probe_count=common_exposure_operator_probe_count,
        common_universe_manifest_digest=common_universe_manifest_digest,
        common_exposure_stage=common_exposure_stage,
        common_exposure_policy_id=common_exposure_policy_id,
        expansion_policy_id=expansion_policy_id,
        eligibility_policy_id=eligibility_policy_id,
        deduplication_policy_id=deduplication_policy_id,
        probe_enumerator_id=probe_enumerator_id,
        event_metadata=event_metadata_out,
    )


def _record_controller_reuse_work_for_records(
    accumulator: "ControllerMeasurementWorkAccumulator | None",
    *,
    runtime: ControllerMeasurementWorkRecordRuntime,
    snapshot: Any | None,
    phase: str,
    event_kind: str,
    records: Sequence[Mapping[str, Any]] | None,
    reuse_key: str,
    reuse_source_event_kind: str,
    source_record_keys: Sequence[str],
    reused_record_keys: Sequence[str],
    depth_value: int | None = None,
    shot_phase: str | None = None,
    scope_qualifiers: Mapping[str, Any] | None = None,
    shortlist_size: int | None = None,
    retained_count: int | None = None,
) -> dict[str, Any] | None:
    if accumulator is None:
        return None
    phase_name = str(phase)
    shot_phase_name = str(shot_phase or phase_name)
    nominal_shots = (
        _controller_phase_shots(snapshot, shot_phase_name, 1)
        if shot_phase_name in {"phase1", "phase2", "phase3"}
        else 1
    )
    if int(nominal_shots) <= 0:
        return None
    group_keys, records_evaluated, records_with_group_keys = (
        _controller_work_group_keys_from_records(records, runtime=runtime)
    )
    return accumulator.record_reuse_event(
        phase=phase_name,
        event_kind=str(event_kind),
        group_keys=group_keys,
        nominal_shots_per_group=int(nominal_shots),
        reused_record_count=int(records_evaluated),
        records_with_group_keys=int(records_with_group_keys),
        depth=None if depth_value is None else int(depth_value),
        shot_phase=shot_phase_name,
        scope_qualifiers=scope_qualifiers,
        shortlist_size=shortlist_size,
        retained_count=retained_count,
        reuse_key=str(reuse_key),
        reuse_source_event_kind=str(reuse_source_event_kind),
        source_record_keys=source_record_keys,
        reused_record_keys=reused_record_keys,
    )


class ControllerMeasurementWorkAccumulator:
    """Low-memory live controller measurement-work accumulator.

    This object is reporting telemetry only.  A measurement group key is a
    Pauli-basis grouping key, not a complete measurement-campaign identity, so
    basis reuse is scoped by phase/event/depth/branch ``work_scope``.  Cache
    reuse fields are diagnostic marginal-work telemetry and are independent of
    the Paper-I total algorithmic-work currency.
    """

    def __init__(
        self,
        *,
        nominal_shots_per_group: int = 1,
        plan_version: str = "phase1_qwc_basis_cover_reuse",
        grouping_mode: str = "qwc_basis_cover_reuse",
        covered_group_shots: Mapping[str, int] | None = None,
        covered_group_shots_by_scope: Mapping[str, Mapping[str, int]] | None = None,
        events: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._default_nominal_shots = int(max(1, nominal_shots_per_group))
        self._plan_version = str(plan_version)
        self._grouping_mode = str(grouping_mode)
        self._covered_group_shots_by_scope: dict[str, dict[str, int]] = {}
        if isinstance(covered_group_shots_by_scope, Mapping):
            for scope, values in covered_group_shots_by_scope.items():
                if not isinstance(values, Mapping):
                    continue
                self._covered_group_shots_by_scope[str(scope)] = {
                    str(k): int(max(1, int(v))) for k, v in dict(values).items()
                }
        if isinstance(covered_group_shots, Mapping) and covered_group_shots:
            legacy_scope = _default_controller_work_scope(
                phase="legacy",
                event_kind="unscoped_cache_import",
                depth=None,
            )
            self._covered_group_shots_by_scope.setdefault(legacy_scope, {}).update(
                {str(k): int(max(1, int(v))) for k, v in dict(covered_group_shots).items()}
            )
        self._events: list[dict[str, Any]] = [dict(event) for event in (events or [])]

    @classmethod
    def from_measurement_cache_template(
        cls, cache: MeasurementCacheAudit | None
    ) -> "ControllerMeasurementWorkAccumulator":
        summary = _cache_summary(cache) if cache is not None else {}
        return cls(
            nominal_shots_per_group=_as_int(summary.get("nominal_shots_per_group"), 1),
            plan_version=str(summary.get("plan_version", "phase1_qwc_basis_cover_reuse")),
            grouping_mode=str(summary.get("grouping_mode", "qwc_basis_cover_reuse")),
        )

    def clone(self) -> "ControllerMeasurementWorkAccumulator":
        return ControllerMeasurementWorkAccumulator(
            nominal_shots_per_group=int(self._default_nominal_shots),
            plan_version=str(self._plan_version),
            grouping_mode=str(self._grouping_mode),
            covered_group_shots_by_scope=deepcopy(self._covered_group_shots_by_scope),
            events=deepcopy(self._events),
        )

    def event_count(self) -> int:
        return int(len(self._events))

    def record_event(
        self,
        *,
        phase: str,
        event_kind: str,
        group_keys: Sequence[str] | None,
        nominal_shots_per_group: int | float | None = None,
        records_evaluated: int = 0,
        records_with_group_keys: int | None = None,
        depth: int | None = None,
        shot_phase: str | None = None,
        source: str = NATIVE_CONTROLLER_WORK_SOURCE,
        work_scope: str | None = None,
        scope_qualifiers: Mapping[str, Any] | None = None,
        candidate_count: int | None = None,
        evaluated_count: int | None = None,
        pre_shortlist_count: int | None = None,
        shortlist_size: int | None = None,
        retained_count: int | None = None,
        rejected_count: int | None = None,
        candidate_work_ledger_scope: str = "event_records_measured_v1",
        probe_role: str | None = None,
        actual_operator_probe_count: int | None = None,
        common_parent_candidate_count: int | None = None,
        common_expanded_candidate_count: int | None = None,
        common_exposure_operator_probe_count: int | None = None,
        common_universe_manifest_digest: str | None = None,
        common_exposure_stage: str = COMMON_EXPOSURE_STAGE,
        common_exposure_policy_id: str = COMMON_EXPOSURE_POLICY_ID,
        expansion_policy_id: str = DEFAULT_EXPANSION_POLICY_ID,
        eligibility_policy_id: str = DEFAULT_ELIGIBILITY_POLICY_ID,
        deduplication_policy_id: str = DEFAULT_DEDUPLICATION_POLICY_ID,
        probe_enumerator_id: str = DEFAULT_PROBE_ENUMERATOR_ID,
        event_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        nominal = int(max(1, round(_as_float(nominal_shots_per_group, float(self._default_nominal_shots)))))
        keys = compress_measurement_group_keys([str(key) for key in (group_keys or []) if str(key) != ""])
        scope = str(
            work_scope
            or _default_controller_work_scope(
                phase=str(phase),
                event_kind=str(event_kind),
                depth=None if depth is None else int(depth),
                scope_qualifiers=scope_qualifiers,
            )
        )
        self._covered_group_shots_by_scope.setdefault(scope, {})
        groups_total = int(len(keys))
        groups_reused = 0
        groups_cache_missed = 0
        groups_topup = 0
        shots_new = 0.0
        shots_reused = 0.0

        for key in keys:
            prior = self._covered_shots_for_required_key(scope, str(key))
            if prior <= 0:
                groups_cache_missed += 1
                shots_new += float(nominal)
            elif prior < nominal:
                groups_topup += 1
                shots_reused += float(prior)
                shots_new += float(nominal - prior)
            else:
                groups_reused += 1
                shots_reused += float(nominal)

        for key in keys:
            self._commit_group_key(scope, str(key), nominal)

        groups_new = int(groups_cache_missed + groups_topup)
        candidate_count_int = int(max(0, candidate_count if candidate_count is not None else records_evaluated))
        evaluated_count_int = int(max(0, evaluated_count if evaluated_count is not None else records_evaluated))
        event = {
            "schema": CONTROLLER_WORK_EVENT_SCHEMA_VERSION,
            "source": str(source),
            "phase": str(phase),
            "shot_phase": str(shot_phase or phase),
            "event_kind": str(event_kind),
            "depth": None if depth is None else int(depth),
            "work_scope": str(scope),
            "work_scope_version": CONTROLLER_WORK_SCOPE_VERSION,
            "scope_qualifiers": _scope_qualifier_payload(scope_qualifiers),
            "records_evaluated": int(max(0, records_evaluated)),
            "records_with_group_keys": int(
                max(0, records_with_group_keys if records_with_group_keys is not None else records_evaluated)
            ),
            "group_key_count": int(groups_total),
            "expanded_measurement_group_probe_count": int(groups_total),
            "expanded_measurement_group_probe_count_total": int(groups_total),
            "nominal_shots_per_group": int(nominal),
            "plan_version": str(self._plan_version),
            "grouping_mode": str(self._grouping_mode),
            "groups_total": int(groups_total),
            "groups_reused": int(groups_reused),
            "groups_cache_missed": int(groups_cache_missed),
            "groups_topup": int(groups_topup),
            "groups_new": int(groups_new),
            "total_groups_new": int(groups_new),
            "shots_total": float(groups_total * nominal),
            "shots_reused": float(shots_reused),
            "shots_new": float(shots_new),
            "total_shots_new": float(shots_new),
            "reuse_count_cost": float(groups_new),
            "counted_in_S_ctrl_proxy": True,
            "decision_live": True,
            "candidate_work_ledger_schema": CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION,
            "candidate_work_ledger_status": CONTROLLER_CANDIDATE_WORK_LEDGER_COMPLETE_STATUS,
            "candidate_work_ledger_scope": str(candidate_work_ledger_scope),
            "candidate_work_event_count": 1,
            "candidate_work_missing_event_count": 0,
            "candidate_count": int(candidate_count_int),
            "evaluated_count": int(evaluated_count_int),
        }
        for key, value in (
            ("pre_shortlist_count", pre_shortlist_count),
            ("shortlist_size", shortlist_size),
            ("retained_count", retained_count),
            ("rejected_count", rejected_count),
        ):
            if value is not None:
                event[key] = int(max(0, value))
        if probe_role not in {None, ""} or actual_operator_probe_count is not None or common_exposure_operator_probe_count is not None:
            event["operator_probe_event_schema"] = OPERATOR_PROBE_EVENT_SCHEMA_VERSION
            event["work_contract_id"] = OPERATOR_PROBE_WORK_CONTRACT_ID
            event["operator_probe_charge_basis"] = OPERATOR_PROBE_CHARGE_BASIS
        if probe_role not in {None, ""}:
            event["probe_role"] = str(probe_role)
        if actual_operator_probe_count is not None:
            event["actual_operator_probe_count"] = int(max(0, actual_operator_probe_count))
            event["actual_evaluated_candidate_count"] = int(evaluated_count_int)
        if common_exposure_operator_probe_count is not None:
            event["common_exposure_operator_probe_count"] = int(max(0, common_exposure_operator_probe_count))
            event["common_exposure_stage"] = str(common_exposure_stage)
            event["common_exposure_policy_id"] = str(common_exposure_policy_id)
            event["expansion_policy_id"] = str(expansion_policy_id)
            event["eligibility_policy_id"] = str(eligibility_policy_id)
            event["deduplication_policy_id"] = str(deduplication_policy_id)
            event["probe_enumerator_id"] = str(probe_enumerator_id)
            if common_parent_candidate_count is not None:
                event["common_parent_candidate_count"] = int(max(0, common_parent_candidate_count))
            if common_expanded_candidate_count is not None:
                event["common_expanded_candidate_count"] = int(max(0, common_expanded_candidate_count))
            if common_universe_manifest_digest not in {None, ""}:
                event["common_universe_manifest_digest"] = str(common_universe_manifest_digest)
        if candidate_count is not None:
            event["method_input_candidate_count"] = int(candidate_count_int)
        if shortlist_size is not None:
            event["method_shortlist_candidate_count"] = int(max(0, shortlist_size))
        if retained_count is not None:
            event["method_retained_candidate_count"] = int(max(0, retained_count))
        if rejected_count is not None:
            event["method_rejected_candidate_count"] = int(max(0, rejected_count))
        if isinstance(event_metadata, Mapping):
            for key, value in event_metadata.items():
                event.setdefault(str(key), value)
        self._events.append(event)
        return dict(event)

    def record_reuse_event(
        self,
        *,
        phase: str,
        event_kind: str,
        group_keys: Sequence[str] | None,
        reused_record_count: int,
        records_with_group_keys: int,
        reuse_key: str,
        reuse_source_event_kind: str,
        source_record_keys: Sequence[str],
        reused_record_keys: Sequence[str],
        nominal_shots_per_group: int | float | None = None,
        depth: int | None = None,
        shot_phase: str | None = None,
        scope_qualifiers: Mapping[str, Any] | None = None,
        shortlist_size: int | None = None,
        retained_count: int | None = None,
    ) -> dict[str, Any]:
        """Record validated reuse with zero incremental probes and shots."""

        reused_count = int(max(0, reused_record_count))
        source_keys = tuple(str(key) for key in source_record_keys)
        reused_keys = tuple(str(key) for key in reused_record_keys)
        source_key_set = set(source_keys)
        reuse_valid = bool(
            str(reuse_key)
            and source_keys
            and reused_keys
            and all(source_keys)
            and all(reused_keys)
            and len(reused_keys) == reused_count
            and set(reused_keys).issubset(source_key_set)
        )
        if not reuse_valid:
            raise ValueError(
                "Validated controller reuse requires nonempty reused record keys "
                "that are an exact subset of the recorded source-event keys."
            )
        source_key_digest = hashlib.sha256(
            json.dumps(
                sorted(source_keys),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        reused_key_digest = hashlib.sha256(
            json.dumps(
                sorted(reused_keys),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        nominal = int(
            max(
                1,
                round(
                    _as_float(
                        nominal_shots_per_group,
                        float(self._default_nominal_shots),
                    )
                ),
            )
        )
        keys = compress_measurement_group_keys(
            [str(key) for key in (group_keys or []) if str(key) != ""]
        )
        event = self.record_event(
            phase=str(phase),
            event_kind=str(event_kind),
            group_keys=[],
            nominal_shots_per_group=int(nominal),
            records_evaluated=int(reused_count),
            records_with_group_keys=int(max(0, records_with_group_keys)),
            depth=None if depth is None else int(depth),
            shot_phase=str(shot_phase or phase),
            scope_qualifiers=scope_qualifiers,
            candidate_count=int(reused_count),
            evaluated_count=0,
            pre_shortlist_count=int(reused_count),
            shortlist_size=shortlist_size,
            retained_count=retained_count,
            rejected_count=int(
                max(
                    0,
                    reused_count
                    - int(
                        retained_count
                        if retained_count is not None
                        else reused_count
                    ),
                )
            ),
            candidate_work_ledger_scope="validated_measurement_reuse_v1",
            probe_role="metric",
            actual_operator_probe_count=0,
        )
        stored = self._events[-1]
        reused_shots = float(len(keys) * nominal)
        stored.update(
            {
                "group_key_count": int(len(keys)),
                "expanded_measurement_group_probe_count": 0,
                "expanded_measurement_group_probe_count_total": 0,
                "groups_total": int(len(keys)),
                "groups_reused": int(len(keys)),
                "groups_cache_missed": 0,
                "groups_topup": 0,
                "groups_new": 0,
                "total_groups_new": 0,
                "shots_total": float(reused_shots),
                "shots_reused": float(reused_shots),
                "shots_new": 0.0,
                "total_shots_new": 0.0,
                "reuse_count_cost": 0.0,
                "reused_operator_probe_count": int(reused_count),
                "measurement_reuse_key": str(reuse_key),
                "measurement_reuse_policy": "exact_full_feature_record_v1",
                "measurement_reuse_validation_status": "exact_match",
                "reuse_source_event_kind": str(reuse_source_event_kind),
                "measurement_reuse_source_record_key_count": int(
                    len(source_keys)
                ),
                "measurement_reuse_record_key_count": int(len(reused_keys)),
                "measurement_reuse_source_record_key_digest": str(
                    source_key_digest
                ),
                "measurement_reuse_record_key_digest": str(reused_key_digest),
                "actual_evaluated_candidate_count": 0,
                "route_a_funnel_reused_probe_role": (
                    "phase2_metric_and_phase3_geometry"
                ),
                "route_a_funnel_reused_record_count": int(reused_count),
                "route_a_funnel_reuse_incremental_operator_probe_count": 0,
            }
        )
        return dict(stored)

    def summary(self, *, include_events: bool = True) -> dict[str, Any]:
        return self.summary_since(0, include_events=include_events)

    def summary_since(self, start_event_index: int, *, include_events: bool = True) -> dict[str, Any]:
        start = int(max(0, min(int(start_event_index), len(self._events))))
        summary = _empty_controller_summary(include_events=include_events)
        summary.update(
            {
                "plan_version": str(self._plan_version),
                "grouping_mode": str(self._grouping_mode),
                "nominal_shots_per_group": int(self._default_nominal_shots),
                "work_scope_version": CONTROLLER_WORK_SCOPE_VERSION,
            }
        )
        for event in self._events[start:]:
            _merge_controller_event(summary, event)
        return _finalize_controller_summary(summary)

    def _covered_shots_for_required_key(self, work_scope: str, required_key: str) -> int:
        best = 0
        scope_cache = self._covered_group_shots_by_scope.setdefault(str(work_scope), {})
        for seen_key, shots in scope_cache.items():
            if measurement_basis_key_covers(str(required_key), str(seen_key)):
                best = max(best, int(shots))
        return int(best)

    def _commit_group_key(self, work_scope: str, key: str, nominal_shots: int) -> None:
        key_s = str(key)
        if key_s == "":
            return
        scope_cache = self._covered_group_shots_by_scope.setdefault(str(work_scope), {})
        prior_cover = self._covered_shots_for_required_key(str(work_scope), key_s)
        if prior_cover >= int(nominal_shots):
            return
        covered = {
            seen
            for seen, shots in scope_cache.items()
            if int(shots) <= int(nominal_shots)
            and measurement_basis_key_covers(str(seen), key_s)
        }
        for seen in covered:
            scope_cache.pop(seen, None)
        scope_cache[key_s] = int(nominal_shots)


def serialize_selector_measurement_cache_stats(
    cache: MeasurementCacheAudit,
    group_keys: Sequence[str] | None,
    *,
    source: str = NATIVE_SOURCE,
    admission_committed: bool = True,
    suppressed_reason: str | None = None,
) -> dict[str, Any]:
    """Serialize admitted-selector measurement stats before cache commit.

    ``cache.estimate`` is read-only, so this is safe to call immediately before
    the existing ``cache.commit`` call.  ``shots_new`` is a nominal proxy whose
    scale is determined by the cache's configured shots per group.
    """

    keys = [str(key) for key in (group_keys or []) if str(key) != ""]
    summary = _cache_summary(cache)
    nominal = _as_float(summary.get("nominal_shots_per_group"), 1.0)
    if not bool(admission_committed):
        return {
            "schema": SCHEMA_VERSION,
            "source": str(source),
            "admission_committed": False,
            "suppressed_reason": None if suppressed_reason is None else str(suppressed_reason),
            "plan_version": str(summary.get("plan_version", "phase1_qwc_basis_cover_reuse")),
            "grouping_mode": str(summary.get("grouping_mode", "qwc_basis_cover_reuse")),
            "nominal_shots_per_group": float(nominal),
            "group_key_count": int(len(keys)),
            "groups_total": 0,
            "groups_reused": 0,
            "groups_new": 0,
            "shots_total": 0.0,
            "shots_reused": 0.0,
            "shots_new": 0.0,
            "reuse_count_cost": 0.0,
        }

    stats = cache.estimate(keys)
    groups_total = int(stats.groups_total)
    shots_total = _as_float(getattr(stats, "shots_total", None), float(groups_total * nominal))
    return {
        "schema": SCHEMA_VERSION,
        "source": str(source),
        "admission_committed": True,
        "suppressed_reason": None,
        "plan_version": str(summary.get("plan_version", "phase1_qwc_basis_cover_reuse")),
        "grouping_mode": str(summary.get("grouping_mode", "qwc_basis_cover_reuse")),
        "nominal_shots_per_group": float(nominal),
        "group_key_count": int(len(keys)),
        "groups_total": groups_total,
        "groups_reused": int(stats.groups_reused),
        "groups_new": int(stats.groups_new),
        "shots_total": float(shots_total),
        "shots_reused": float(stats.shots_reused),
        "shots_new": float(stats.shots_new),
        "reuse_count_cost": float(stats.reuse_count_cost),
    }


def history_selector_measurement_stats(history_row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return native selector stats for a history row, with legacy fallback.

    The manuscript proxy is admitted selector nominal shots.  Scored-surface
    diagnostics are intentionally not considered here.
    """

    if not isinstance(history_row, Mapping):
        return None
    native = history_row.get("selector_measurement_cache_stats")
    native_has_expected_payload = (
        isinstance(native, Mapping)
        and (
            any(key in native for key in ("groups_new", "shots_new"))
            or not bool(native.get("admission_committed", True))
        )
    )
    if native_has_expected_payload:
        rec = dict(native)
        rec.setdefault("schema", SCHEMA_VERSION)
        rec.setdefault("source", NATIVE_SOURCE)
        rec["source_kind"] = "native"
        if not bool(rec.get("admission_committed", True)):
            rec["groups_new"] = 0
            rec["shots_new"] = 0.0
            rec["reuse_count_cost"] = 0.0
        return rec

    legacy = history_row.get("measurement_cache_stats")
    if not isinstance(legacy, Mapping):
        return None
    groups_new = _as_int(legacy.get("groups_new"), 0)
    shots_new = _as_float(legacy.get("shots_new"), 0.0)
    return {
        "schema": SCHEMA_VERSION,
        "source": LEGACY_HISTORY_SOURCE,
        "source_kind": "legacy_history",
        "admission_committed": True,
        "suppressed_reason": None,
        "plan_version": str(legacy.get("plan_version", "legacy_history")),
        "grouping_mode": str(legacy.get("grouping_mode", "legacy_history")),
        "nominal_shots_per_group": _as_float(legacy.get("nominal_shots_per_group"), 1.0),
        "group_key_count": _as_int(legacy.get("group_key_count", legacy.get("groups_total", 0)), 0),
        "groups_total": _as_int(legacy.get("groups_total"), 0),
        "groups_reused": _as_int(legacy.get("groups_reused"), 0),
        "groups_new": int(groups_new),
        "shots_total": _as_float(legacy.get("shots_total"), shots_new),
        "shots_reused": _as_float(legacy.get("shots_reused"), 0.0),
        "shots_new": float(shots_new),
        "reuse_count_cost": _as_float(legacy.get("reuse_count_cost"), 0.0),
    }


def _finalize_controller_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    rec = dict(summary)
    rec.setdefault("schema", CONTROLLER_WORK_SCHEMA_VERSION)
    rec.setdefault("source", NATIVE_CONTROLLER_WORK_SOURCE)
    rec.setdefault("source_kind", "native_controller_work")
    rec.setdefault("legacy_fallback_used", False)
    rec.setdefault("legacy_equivalent_to_controller_work", True)
    by_scope = rec.get("by_scope")
    if isinstance(by_scope, Mapping):
        rec["work_scope_count"] = int(len(by_scope))
        if by_scope or "work_scope_version" in rec:
            rec.setdefault("work_scope_version", CONTROLLER_WORK_SCOPE_VERSION)
    else:
        rec.setdefault("work_scope_count", 0)
    groups_new = _as_float(rec.get("total_groups_new", rec.get("groups_new", 0.0)), 0.0)
    shots_new = _as_float(rec.get("total_shots_new", rec.get("shots_new", 0.0)), 0.0)
    expanded_probe_count = _as_float(
        rec.get(
            "expanded_measurement_group_probe_count_total",
            rec.get("expanded_measurement_group_probe_count", rec.get("groups_total", 0.0)),
        ),
        0.0,
    )
    rec["groups_new"] = groups_new
    rec["total_groups_new"] = groups_new
    rec["expanded_measurement_group_probe_count"] = expanded_probe_count
    rec["expanded_measurement_group_probe_count_total"] = expanded_probe_count
    rec["shots_new"] = shots_new
    rec["total_shots_new"] = shots_new
    validation = _finalized_numeric_validation(rec)
    rec["numeric_validation"] = validation
    rec["controller_numeric_validation"] = validation
    rec["controller_numeric_validation_status"] = validation["status"]
    rec["paper_i_controller_numeric_validation_status"] = validation["paper_i_table_work_status"]
    rec.setdefault("controller_group_proxy", float(groups_new))
    rec.setdefault("controller_shot_proxy", float(shots_new))
    rec.setdefault("controller_reuse_proxy", _as_float(rec.get("reuse_count_cost"), 0.0))
    rec.setdefault("controller_proxy_source", str(rec.get("source_kind", "native_controller_work")))
    rec.setdefault(
        "controller_proxy_legacy_fallback_used",
        bool(rec.get("legacy_fallback_used", False)),
    )
    rec.setdefault("candidate_work_ledger_schema", CONTROLLER_CANDIDATE_WORK_LEDGER_SCHEMA_VERSION)
    candidate_events = _as_int(rec.get("candidate_work_event_count"), 0)
    candidate_missing = _as_int(rec.get("candidate_work_missing_event_count"), 0)
    work_events = _as_int(rec.get("events_count"), 0)
    has_controller_work = bool(
        work_events > 0
        or _as_float(rec.get("records_evaluated"), 0.0) > 0.0
        or _as_float(rec.get("records_with_group_keys"), 0.0) > 0.0
        or bool(rec.get("by_phase") or rec.get("per_phase"))
    )
    if candidate_events > 0 and candidate_missing == 0 and (work_events <= 0 or candidate_events >= work_events):
        rec["candidate_work_ledger_status"] = CONTROLLER_CANDIDATE_WORK_LEDGER_COMPLETE_STATUS
    elif not has_controller_work:
        rec["candidate_work_ledger_status"] = "no_controller_work_events"
    else:
        rec["candidate_work_ledger_status"] = CONTROLLER_CANDIDATE_WORK_LEDGER_MISSING_STATUS
        if candidate_missing <= 0:
            rec["candidate_work_missing_event_count"] = int(max(1, work_events))
    scopes = rec.get("candidate_work_ledger_scopes")
    if isinstance(scopes, Mapping) and scopes:
        rec["candidate_work_ledger_scope"] = (
            next(iter(scopes.keys())) if len(scopes) == 1 else "mixed"
        )
    return rec


def validate_controller_proxy_for_shot_objective(proxy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a controller proxy is native live-controller work.

    The Optuna shot-cost objective is only meaningful for controller decision
    work measured through the live controller path.  Legacy admitted-selector
    fallbacks are still useful telemetry, but they must not activate the shot
    objective term.
    """

    raw = proxy if isinstance(proxy, Mapping) else {}
    raw_source = str(raw.get("source", "") or "")
    raw_source_kind = str(raw.get("source_kind", raw.get("controller_proxy_source", "")) or "")
    raw_controller_proxy_source = str(raw.get("controller_proxy_source", raw_source_kind) or raw_source_kind)
    raw_legacy_fallback = bool(
        raw.get("legacy_fallback_used", raw.get("controller_proxy_legacy_fallback_used", False))
    )
    raw_shot_values = [
        raw.get(key)
        for key in ("total_shots_new", "shots_new", "controller_shot_proxy")
        if key in raw
    ]
    has_explicit_numeric_shot_proxy = False
    for value in raw_shot_values:
        try:
            has_explicit_numeric_shot_proxy = math.isfinite(float(value))
        except Exception:
            has_explicit_numeric_shot_proxy = False
        if has_explicit_numeric_shot_proxy:
            break
    has_explicit_native_source = (
        raw_source == NATIVE_CONTROLLER_WORK_SOURCE
        or raw_source == "native_controller_work"
        or raw_source_kind == "native_controller_work"
        or raw_controller_proxy_source == "native_controller_work"
    )

    rec = _finalize_controller_summary(raw)
    source = str(rec.get("source", "missing") or "missing")
    source_kind = str(rec.get("source_kind", rec.get("controller_proxy_source", source)) or "missing")
    controller_proxy_source = str(rec.get("controller_proxy_source", source_kind) or source_kind)
    legacy_fallback = bool(
        rec.get("legacy_fallback_used", rec.get("controller_proxy_legacy_fallback_used", False))
    ) or raw_legacy_fallback
    events_count = _as_int(raw.get("events_count", rec.get("events_count")), 0)
    native_row_count = _as_int(raw.get("native_row_count", rec.get("native_row_count")), 0)
    history_row_count = _as_int(raw.get("history_row_count", rec.get("history_row_count")), 0)
    has_native_work = bool(events_count > 0 or native_row_count > 0 or (history_row_count > 0 and not legacy_fallback))

    valid = bool(
        has_explicit_native_source
        and has_native_work
        and has_explicit_numeric_shot_proxy
        and not legacy_fallback
    )
    if valid:
        reason = "valid_native_controller_work"
    elif legacy_fallback:
        reason = "legacy_fallback"
    elif not has_explicit_native_source:
        reason = "not_native_controller_work"
    elif not has_native_work:
        reason = "no_native_controller_work"
    elif not has_explicit_numeric_shot_proxy:
        reason = "missing_shot_proxy"
    else:  # pragma: no cover - defensive completeness
        reason = "invalid_controller_proxy"

    return {
        "schema": "controller_measurement_proxy_validation_v1",
        "valid": bool(valid),
        "reason": str(reason),
        "source": source,
        "source_kind": source_kind,
        "controller_proxy_source": controller_proxy_source,
        "legacy_fallback_used": bool(legacy_fallback),
        "events_count": int(events_count),
        "native_row_count": int(native_row_count),
        "legacy_row_count": int(_as_int(rec.get("legacy_row_count"), 0)),
        "history_row_count": int(history_row_count),
        "has_shot_proxy": bool(has_explicit_numeric_shot_proxy),
        "total_shots_new": _as_float(rec.get("total_shots_new", rec.get("shots_new", 0.0)), 0.0),
        "total_groups_new": _as_float(rec.get("total_groups_new", rec.get("groups_new", 0.0)), 0.0),
    }


def _controller_summary_from_legacy_selector(stats: Mapping[str, Any]) -> dict[str, Any]:
    groups = _as_float(stats.get("groups_new"), 0.0)
    shots = _as_float(stats.get("shots_new"), 0.0)
    reuse = _as_float(stats.get("reuse_count_cost"), 0.0)
    return _finalize_controller_summary(
        {
            "schema": CONTROLLER_WORK_SCHEMA_VERSION,
            "source": LEGACY_CONTROLLER_WORK_FALLBACK_SOURCE,
            "source_kind": "legacy_admitted_selector",
            "legacy_fallback_used": True,
            "legacy_equivalent_to_controller_work": False,
            "events_count": 1,
            "records_evaluated": 1 if groups > 0 or shots > 0 else 0,
            "records_with_group_keys": 1 if groups > 0 or shots > 0 else 0,
            "groups_total": _as_float(stats.get("groups_total"), groups),
            "groups_reused": _as_float(stats.get("groups_reused"), 0.0),
            "groups_cache_missed": groups,
            "groups_topup": 0,
            "groups_new": groups,
            "total_groups_new": groups,
            "shots_total": _as_float(stats.get("shots_total"), shots),
            "shots_reused": _as_float(stats.get("shots_reused"), 0.0),
            "shots_new": shots,
            "total_shots_new": shots,
            "reuse_count_cost": reuse,
            "per_phase": {},
            "by_phase": {},
            "events": [],
        }
    )


def history_controller_measurement_work_stats(history_row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return live controller work stats for a history row, with legacy fallback.

    Native controller work counts live decision-evaluated records/events.  The
    fallback intentionally uses admitted selector/cache telemetry only for old
    artifacts and marks it as not equivalent to native controller work.
    """

    if not isinstance(history_row, Mapping):
        return None
    native = history_row.get("controller_measurement_work_proxy")
    if isinstance(native, Mapping) and (
        any(key in native for key in ("total_shots_new", "shots_new", "total_groups_new", "groups_new"))
        or str(native.get("schema", "")) == CONTROLLER_WORK_SCHEMA_VERSION
    ):
        finalized = _finalize_controller_summary(native)
        return _apply_phase3_batch_summary_supplement(history_row, finalized)
    selector = history_selector_measurement_stats(history_row)
    if selector is None:
        return None
    return _controller_summary_from_legacy_selector(selector)


def _history_rows(adapt_payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    payload = adapt_payload.get("adapt_vqe", adapt_payload) if isinstance(adapt_payload, Mapping) else {}
    if not isinstance(payload, Mapping):
        return ()
    rows = payload.get("history", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def controller_proxy_from_history_rows(history_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum live controller measurement-work proxy from ADAPT history rows."""

    summary = _empty_controller_summary(include_events=False)
    native_rows = 0
    fallback_rows = 0
    skipped_rows = 0
    scoped_rows = 0
    for row in history_rows:
        stats = history_controller_measurement_work_stats(row)
        if stats is None:
            skipped_rows += 1
            continue
        if bool(stats.get("legacy_fallback_used", False)):
            fallback_rows += 1
        else:
            native_rows += 1
        if stats.get("work_scope_version") == CONTROLLER_WORK_SCOPE_VERSION or bool(stats.get("by_scope")):
            scoped_rows += 1
        event_like = {
            "phase": "run_total",
            "events_count": stats.get("events_count", 1),
            "records_evaluated": stats.get("records_evaluated", 0),
            "records_with_group_keys": stats.get("records_with_group_keys"),
            "groups_total": stats.get("groups_total", 0),
            "groups_reused": stats.get("groups_reused", 0),
            "groups_cache_missed": stats.get("groups_cache_missed", stats.get("groups_new", 0)),
            "groups_topup": stats.get("groups_topup", 0),
            "groups_new": stats.get("total_groups_new", stats.get("groups_new", 0)),
            "total_groups_new": stats.get("total_groups_new", stats.get("groups_new", 0)),
            "expanded_measurement_group_probe_count": stats.get(
                "expanded_measurement_group_probe_count_total",
                stats.get("expanded_measurement_group_probe_count", stats.get("groups_total", 0)),
            ),
            "expanded_measurement_group_probe_count_total": stats.get(
                "expanded_measurement_group_probe_count_total",
                stats.get("expanded_measurement_group_probe_count", stats.get("groups_total", 0)),
            ),
            "shots_total": stats.get("shots_total", 0),
            "shots_reused": stats.get("shots_reused", 0),
            "shots_new": stats.get("total_shots_new", stats.get("shots_new", 0)),
            "total_shots_new": stats.get("total_shots_new", stats.get("shots_new", 0)),
            "reuse_count_cost": stats.get("reuse_count_cost", 0),
            "candidate_work_ledger_schema": stats.get("candidate_work_ledger_schema"),
            "candidate_work_ledger_status": stats.get("candidate_work_ledger_status"),
            "candidate_work_ledger_scope": stats.get("candidate_work_ledger_scope"),
            "candidate_work_ledger_scopes": stats.get("candidate_work_ledger_scopes"),
            "candidate_work_event_count": stats.get("candidate_work_event_count"),
            "candidate_work_missing_event_count": stats.get("candidate_work_missing_event_count"),
            "candidate_count_total": stats.get("candidate_count_total", stats.get("candidate_count", 0)),
            "evaluated_count_total": stats.get("evaluated_count_total", stats.get("evaluated_count", 0)),
            "pre_shortlist_count_total": stats.get(
                "pre_shortlist_count_total",
                stats.get("pre_shortlist_count", 0),
            ),
            "shortlist_size_total": stats.get("shortlist_size_total", stats.get("shortlist_size", 0)),
            "retained_count_total": stats.get("retained_count_total", stats.get("retained_count", 0)),
            "rejected_count_total": stats.get("rejected_count_total", stats.get("rejected_count", 0)),
        }
        _merge_controller_event_no_events(summary, event_like)
        phase_map = summary.setdefault("per_phase", {})
        if not isinstance(phase_map, dict):
            phase_map = {}
            summary["per_phase"] = phase_map
        _merge_nested_controller_summaries(
            phase_map,
            stats.get("by_phase", stats.get("per_phase")),
        )
        summary["by_phase"] = summary["per_phase"]
        scope_map = summary.setdefault("by_scope", {})
        if not isinstance(scope_map, dict):
            scope_map = {}
            summary["by_scope"] = scope_map
        _merge_nested_controller_summaries(scope_map, stats.get("by_scope"))
        summary["work_scope_count"] = int(len(scope_map))
    source = "missing"
    if native_rows and fallback_rows:
        source = "mixed_native_controller_legacy_admitted_selector"
    elif native_rows:
        source = "native_controller_work"
    elif fallback_rows:
        source = "legacy_admitted_selector"
    summary.update(
        {
            "source": source,
            "source_kind": source,
            "legacy_fallback_used": bool(fallback_rows > 0),
            "legacy_equivalent_to_controller_work": bool(fallback_rows == 0),
            "controller_proxy_source": source,
            "controller_proxy_legacy_fallback_used": bool(fallback_rows > 0),
            "history_row_count": int(native_rows + fallback_rows),
            "native_row_count": int(native_rows),
            "legacy_row_count": int(fallback_rows),
            "skipped_row_count": int(skipped_rows),
            "scoped_row_count": int(scoped_rows),
        }
    )
    if int(scoped_rows) <= 0:
        summary.pop("work_scope_version", None)
        summary["work_scope_count"] = 0
    return _finalize_controller_summary(summary)


def controller_proxy_from_adapt_payload(adapt_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return live controller measurement-work proxy from an ADAPT payload."""

    payload = adapt_payload.get("adapt_vqe", adapt_payload) if isinstance(adapt_payload, Mapping) else {}
    if isinstance(payload, Mapping):
        native_summary = payload.get("controller_measurement_work_summary")
        if isinstance(native_summary, Mapping) and (
            "total_shots_new" in native_summary or "shots_new" in native_summary
        ):
            return _finalize_controller_summary(native_summary)
    return controller_proxy_from_history_rows(_history_rows(adapt_payload))


def selector_proxy_from_adapt_payload(adapt_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sum admitted selector nominal group/shot proxy from an ADAPT payload."""

    groups = 0.0
    shots = 0.0
    reuse = 0.0
    rows_seen = 0
    native_rows = 0
    legacy_rows = 0
    skipped_rows = 0
    source = "missing"
    for row in _history_rows(adapt_payload):
        stats = history_selector_measurement_stats(row)
        if stats is None:
            skipped_rows += 1
            continue
        rows_seen += 1
        kind = str(stats.get("source_kind", "legacy_history"))
        if kind == "native":
            native_rows += 1
        else:
            legacy_rows += 1
        groups += _as_float(stats.get("groups_new"), 0.0)
        shots += _as_float(stats.get("shots_new"), 0.0)
        reuse += _as_float(stats.get("reuse_count_cost"), 0.0)
    if native_rows and legacy_rows:
        source = "mixed_native_legacy_history"
    elif native_rows:
        source = "native"
    elif legacy_rows:
        source = "legacy_history"
    return {
        "selector_proxy_version": SCHEMA_VERSION,
        "selector_group_proxy": float(groups),
        "selector_shot_proxy": float(shots),
        "selector_reuse_proxy": float(reuse),
        "selector_proxy_source": source,
        "selector_proxy_legacy_fallback_used": bool(legacy_rows > 0),
        "history_row_count": int(rows_seen),
        "native_row_count": int(native_rows),
        "legacy_row_count": int(legacy_rows),
        "skipped_row_count": int(skipped_rows),
    }
