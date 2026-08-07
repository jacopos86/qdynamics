"""Phase shortlist helpers for static ADAPT.

This module owns shortlist selection and legacy shortlist-hook compatibility.
It does not own candidate scoring, controller updates, or admission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    phase_shortlist_records,
    shortlist_records,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.algebraic_metadata import (
    LANES_PHASE1,
    LANE_MIX,
    lane_phase1_shortlist_records,
    lane_phase2_health_shortlist_records,
    lane_quota_pressure_budgets,
)
from pipelines.static_adapt.controller_phase_state import _record_controller_snapshot
from pipelines.static_adapt.route_a_shortlists import (
    ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR,
    expand_selected_identities,
    identity_population,
    macro_operator_identity,
)

__all__ = [
    "PhaseShortlistRuntime",
    "_notify_legacy_shortlist_hook",
    "_phase1_eval_payload_from_records",
    "_phase1_lane_shortlist_with_legacy_hook",
    "_phase1_record_score_value",
    "_phase1_shortlist_score_key",
    "_phase2_lane_health_shortlist_with_legacy_hook",
    "_phase3_tie_beam_selection_pool",
    "_phase_shortlist_with_legacy_hook",
    "_positive_phase3_selector_records",
    "_record_algebraic_shortlist_runtime",
    "_selection_pool_from_shortlist",
    "_selection_record_key",
]


@dataclass(frozen=True)
class PhaseShortlistRuntime:
    phase2_score_cfg: FullScoreConfig
    algebraic_lane_policy_active: bool
    algebraic_metadata_summary: MutableMapping[str, Any]
    algebraic_phase1_lane_quota_pressure: float
    algebraic_phase2_lane_quota_pressure: float
    algebraic_phase2_lane_rel_threshold: float
    feature_updater: Callable[[Any, Mapping[str, Any]], Any]
    shortlist_lane_policy_active: bool | None = None
    shortlist_lane_route: str = "algebraic"
    shortlist_lane_key: str = "algebraic_lane"
    shortlist_lanes: tuple[str, ...] = LANES_PHASE1
    shortlist_fallback_lane: str = LANE_MIX
    shortlist_lane_health_key_prefix: str = "algebraic"
    shortlist_lane_summary: MutableMapping[str, Any] | None = None
    physical_operator_identity_caps_enabled: bool = True


def _shortlist_lane_policy_active(runtime: PhaseShortlistRuntime) -> bool:
    if runtime.shortlist_lane_policy_active is not None:
        return bool(runtime.shortlist_lane_policy_active)
    return bool(runtime.algebraic_lane_policy_active)


def _physical_operator_identity_caps_active(runtime: PhaseShortlistRuntime) -> bool:
    return bool(
        str(runtime.shortlist_lane_route) == "physical_operator_type"
        and runtime.physical_operator_identity_caps_enabled
    )


def _macro_identity_selection_population(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    tie_break_score_key: str | None,
):
    return identity_population(
        records,
        identity_key=macro_operator_identity,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )


def _expand_physical_operator_selection(
    population: Any,
    selected_representatives: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    shortlist_flag: str | None,
) -> list[dict[str, Any]]:
    expanded = expand_selected_identities(
        population,
        selected_representatives,
        shortlist_flag=shortlist_flag,
        shortlist_unit=ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR,
        feature_updater=runtime.feature_updater,
    )
    health_prefix = str(runtime.shortlist_lane_health_key_prefix)
    health_keys = (
        f"{health_prefix}_lane_health",
        f"{health_prefix}_lane_relative_health",
        f"{health_prefix}_lane_live",
    )
    representatives_by_identity = {
        str(record.get("route_a_shortlist_identity", "")): record
        for record in selected_representatives
    }
    for row in expanded:
        representative = representatives_by_identity.get(
            str(row.get("route_a_shortlist_identity", "")),
            {},
        )
        updates = {
            key: representative[key]
            for key in health_keys
            if key in representative
        }
        if not updates:
            continue
        row.update(updates)
        feature = row.get("feature")
        if feature is not None:
            row["feature"] = runtime.feature_updater(feature, updates)
    return expanded


def _phase1_shortlist_score_key() -> str:
    return "phase1_active_score"


def _phase1_record_score_value(
    rec: Mapping[str, Any],
    *,
    default: float = float("-inf"),
) -> float:
    for key in (_phase1_shortlist_score_key(), "cheap_score", "simple_score"):
        value = rec.get(key)
        if value is None:
            continue
        return float(value)
    return float(default)


def _phase1_eval_payload_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    append_position_value: int,
) -> dict[str, Any]:
    best_score = float("-inf")
    best_idx = -1
    best_position = int(append_position_value)
    best_feat: dict[str, Any] | None = None
    append_best_score = float("-inf")
    append_best_g_lcb = 0.0
    append_best_family = ""
    best_non_append_score = float("-inf")
    best_non_append_g_lcb = 0.0
    records_list = [dict(rec) for rec in records]
    for rec in records_list:
        score_val = _phase1_record_score_value(rec)
        pos = int(rec.get("position_id", append_position_value))
        idx = int(rec.get("candidate_pool_index", -1))
        feat_obj = rec.get("feature")
        feat_dict = (
            dict(feat_obj.__dict__)
            if isinstance(feat_obj, CandidateFeatures)
            else (dict(feat_obj) if isinstance(feat_obj, Mapping) else {})
        )
        g_lcb_val = float(feat_dict.get("g_hw_lcb", feat_dict.get("g_lcb", 0.0)))
        if pos == int(append_position_value) and score_val > append_best_score:
            append_best_score = float(score_val)
            append_best_g_lcb = float(g_lcb_val)
            append_best_family = str(feat_dict.get("candidate_family", ""))
        if pos != int(append_position_value) and score_val > best_non_append_score:
            best_non_append_score = float(score_val)
            best_non_append_g_lcb = float(g_lcb_val)
        if score_val > best_score:
            best_score = float(score_val)
            best_idx = int(idx)
            best_position = int(pos)
            best_feat = dict(feat_dict)
    return {
        "best_score": float(best_score),
        "best_idx": int(best_idx),
        "best_position": int(best_position),
        "best_feat": (dict(best_feat) if isinstance(best_feat, dict) else None),
        "append_best_score": float(append_best_score),
        "append_best_g_lcb": float(append_best_g_lcb),
        "append_best_family": str(append_best_family),
        "best_non_append_score": float(best_non_append_score),
        "best_non_append_g_lcb": float(best_non_append_g_lcb),
        "records": list(records_list),
    }


def _notify_legacy_shortlist_hook(
    records: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    tie_break_score_key: str | None = None,
) -> None:
    records_list = [dict(rec) for rec in records]
    if not records_list:
        return
    legacy_cfg = replace(
        runtime.phase2_score_cfg,
        shortlist_fraction=1.0,
        shortlist_size=max(1, len(records_list)),
    )
    shortlist_records(
        records_list,
        cfg=legacy_cfg,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )


def _phase_shortlist_with_legacy_hook(
    records: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
) -> list[dict[str, Any]]:
    records_list = [dict(rec) for rec in records]
    if int(cap) <= 0:
        return []
    _notify_legacy_shortlist_hook(
        records_list,
        runtime=runtime,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )
    shortlisted = phase_shortlist_records(
        records_list,
        score_key=score_key,
        threshold=threshold,
        cap=cap,
        frontier_ratio=frontier_ratio,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
    )
    if shortlisted:
        return shortlisted
    if not records_list:
        return []
    return phase_shortlist_records(
        records_list,
        score_key=score_key,
        threshold=float("-inf"),
        cap=1,
        frontier_ratio=0.0,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
    )


def _record_algebraic_shortlist_runtime(
    phase: str,
    *,
    runtime: PhaseShortlistRuntime,
    lane_budgets: Mapping[str, int],
    shortlist_size: int,
) -> None:
    summary = (
        runtime.shortlist_lane_summary
        if runtime.shortlist_lane_summary is not None
        else runtime.algebraic_metadata_summary
    )
    runtime_payload = summary.setdefault("shortlist_runtime", {})
    normalized = {
        str(lane): int(max(0, lane_budgets.get(str(lane), 0)))
        for lane in runtime.shortlist_lanes
    }
    if str(runtime.shortlist_lane_route) != "algebraic":
        runtime_payload["lane_route"] = str(runtime.shortlist_lane_route)
        runtime_payload["lane_key"] = str(runtime.shortlist_lane_key)
        runtime_payload["lanes"] = [str(lane) for lane in runtime.shortlist_lanes]
    runtime_payload[f"{phase}_last_lane_budgets"] = normalized
    runtime_payload[f"{phase}_last_budget_target"] = int(sum(normalized.values()))
    runtime_payload[f"{phase}_last_shortlist_size"] = int(max(0, shortlist_size))


def _record_identity_shortlist_runtime(
    phase: str,
    *,
    runtime: PhaseShortlistRuntime,
    input_identity_count: int,
    selected_identity_count: int,
    expanded_record_count: int,
) -> None:
    summary = (
        runtime.shortlist_lane_summary
        if runtime.shortlist_lane_summary is not None
        else runtime.algebraic_metadata_summary
    )
    runtime_payload = summary.setdefault("shortlist_runtime", {})
    runtime_payload[f"{phase}_shortlist_unit"] = ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR
    runtime_payload[f"{phase}_input_identity_count"] = int(input_identity_count)
    runtime_payload[f"{phase}_selected_identity_count"] = int(selected_identity_count)
    runtime_payload[f"{phase}_expanded_position_record_count"] = int(
        expanded_record_count
    )


def _phase1_lane_shortlist_with_legacy_hook(
    records: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
) -> list[dict[str, Any]]:
    records_list = [dict(rec) for rec in records]
    if not _shortlist_lane_policy_active(runtime):
        return _phase_shortlist_with_legacy_hook(
            records_list,
            runtime=runtime,
            score_key=score_key,
            threshold=threshold,
            cap=cap,
            frontier_ratio=frontier_ratio,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )
    if int(cap) <= 0:
        return []
    _notify_legacy_shortlist_hook(
        records_list,
        runtime=runtime,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )
    identity_population_value = None
    selection_records = records_list
    selection_threshold = float(threshold)
    selection_frontier_ratio = float(frontier_ratio)
    if _physical_operator_identity_caps_active(runtime):
        identity_population_value = _macro_identity_selection_population(
            records_list,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        )
        selection_records = [
            dict(record) for record in identity_population_value.representatives
        ]
        selection_threshold = float("-inf")
        selection_frontier_ratio = 0.0
    lane_budgets = lane_quota_pressure_budgets(
        selection_records,
        score_key=score_key,
        threshold=selection_threshold,
        cap=cap,
        pressure=float(runtime.algebraic_phase1_lane_quota_pressure),
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        tie_break_score_key=tie_break_score_key,
    )
    shortlisted = lane_phase1_shortlist_records(
        selection_records,
        score_key=score_key,
        threshold=selection_threshold,
        cap=cap,
        frontier_ratio=selection_frontier_ratio,
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_budgets=lane_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=(
            None
            if identity_population_value is not None
            else runtime.feature_updater
        ),
    )
    if shortlisted:
        if identity_population_value is not None:
            selected_identity_count = int(len(shortlisted))
            shortlisted = _expand_physical_operator_selection(
                identity_population_value,
                shortlisted,
                runtime=runtime,
                shortlist_flag=shortlist_flag,
            )
            _record_identity_shortlist_runtime(
                "phase1",
                runtime=runtime,
                input_identity_count=int(identity_population_value.identity_count),
                selected_identity_count=int(selected_identity_count),
                expanded_record_count=int(len(shortlisted)),
            )
        _record_algebraic_shortlist_runtime(
            "phase1",
            runtime=runtime,
            lane_budgets=lane_budgets,
            shortlist_size=(
                selected_identity_count
                if identity_population_value is not None
                else len(shortlisted)
            ),
        )
        return shortlisted
    fallback_budgets = lane_quota_pressure_budgets(
        records_list,
        score_key=score_key,
        threshold=float("-inf"),
        cap=1,
        pressure=float(runtime.algebraic_phase1_lane_quota_pressure),
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        tie_break_score_key=tie_break_score_key,
    )
    fallback = lane_phase1_shortlist_records(
        records_list,
        score_key=score_key,
        threshold=float("-inf"),
        cap=1,
        frontier_ratio=0.0,
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_budgets=fallback_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=runtime.feature_updater,
    )
    _record_algebraic_shortlist_runtime(
        "phase1",
        runtime=runtime,
        lane_budgets=fallback_budgets,
        shortlist_size=len(fallback),
    )
    return fallback


def _phase2_lane_health_shortlist_with_legacy_hook(
    records: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
) -> list[dict[str, Any]]:
    records_list = [dict(rec) for rec in records]
    if not _shortlist_lane_policy_active(runtime):
        return _phase_shortlist_with_legacy_hook(
            records_list,
            runtime=runtime,
            score_key=score_key,
            threshold=threshold,
            cap=cap,
            frontier_ratio=frontier_ratio,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )
    if int(cap) <= 0:
        return []
    _notify_legacy_shortlist_hook(
        records_list,
        runtime=runtime,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )
    identity_population_value = None
    selection_records = records_list
    selection_threshold = float(threshold)
    selection_frontier_ratio = float(frontier_ratio)
    lane_relative_threshold = float(runtime.algebraic_phase2_lane_rel_threshold)
    if _physical_operator_identity_caps_active(runtime):
        identity_population_value = _macro_identity_selection_population(
            records_list,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        )
        selection_records = [
            dict(record) for record in identity_population_value.representatives
        ]
        selection_threshold = float("-inf")
        selection_frontier_ratio = 0.0
        lane_relative_threshold = 0.0
    lane_budgets = lane_quota_pressure_budgets(
        selection_records,
        score_key=score_key,
        threshold=selection_threshold,
        cap=cap,
        pressure=float(runtime.algebraic_phase2_lane_quota_pressure),
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_rel_threshold=float(lane_relative_threshold),
        tie_break_score_key=tie_break_score_key,
    )
    shortlisted = lane_phase2_health_shortlist_records(
        selection_records,
        score_key=score_key,
        threshold=selection_threshold,
        cap=cap,
        frontier_ratio=selection_frontier_ratio,
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_rel_threshold=float(lane_relative_threshold),
        lane_budgets=lane_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=(
            None
            if identity_population_value is not None
            else runtime.feature_updater
        ),
        health_key_prefix=str(runtime.shortlist_lane_health_key_prefix),
    )
    if shortlisted:
        if identity_population_value is not None:
            selected_identity_count = int(len(shortlisted))
            shortlisted = _expand_physical_operator_selection(
                identity_population_value,
                shortlisted,
                runtime=runtime,
                shortlist_flag=shortlist_flag,
            )
            _record_identity_shortlist_runtime(
                "phase2",
                runtime=runtime,
                input_identity_count=int(identity_population_value.identity_count),
                selected_identity_count=int(selected_identity_count),
                expanded_record_count=int(len(shortlisted)),
            )
        _record_algebraic_shortlist_runtime(
            "phase2",
            runtime=runtime,
            lane_budgets=lane_budgets,
            shortlist_size=(
                selected_identity_count
                if identity_population_value is not None
                else len(shortlisted)
            ),
        )
        return shortlisted
    fallback_budgets = lane_quota_pressure_budgets(
        records_list,
        score_key=score_key,
        threshold=float("-inf"),
        cap=1,
        pressure=float(runtime.algebraic_phase2_lane_quota_pressure),
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_rel_threshold=0.0,
        tie_break_score_key=tie_break_score_key,
    )
    fallback = lane_phase2_health_shortlist_records(
        records_list,
        score_key=score_key,
        threshold=float("-inf"),
        cap=1,
        frontier_ratio=0.0,
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        lane_rel_threshold=0.0,
        lane_budgets=fallback_budgets,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
        feature_updater=runtime.feature_updater,
        health_key_prefix=str(runtime.shortlist_lane_health_key_prefix),
    )
    _record_algebraic_shortlist_runtime(
        "phase2",
        runtime=runtime,
        lane_budgets=fallback_budgets,
        shortlist_size=len(fallback),
    )
    return fallback


def _selection_record_key(rec: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(rec.get("candidate_label") or getattr(rec.get("candidate_term"), "label", "")),
        int(rec.get("candidate_pool_index", -1)),
        int(rec.get("position_id", -1)),
    )


def _positive_phase3_selector_records(
    records: Sequence[Mapping[str, Any]],
    *,
    selector_score_key: str,
) -> list[dict[str, Any]]:
    return [
        dict(rec)
        for rec in records
        if float(rec.get(str(selector_score_key), float("-inf"))) > 0.0
    ]


def _selection_pool_from_shortlist(
    shortlist_records_in: Sequence[Mapping[str, Any]],
    full_records_in: Sequence[Mapping[str, Any]],
    *,
    selector_score_key: str,
    record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    out = [dict(rec) for rec in shortlist_records_in]
    seen = {_selection_record_key(rec) for rec in out}
    positive_full_records = _positive_phase3_selector_records(
        full_records_in,
        selector_score_key=selector_score_key,
    )
    fallback_candidates = (
        positive_full_records
        if positive_full_records
        else [dict(rec) for rec in full_records_in[:1]]
    )
    for rec in fallback_candidates[:1]:
        rec_key = _selection_record_key(rec)
        if rec_key not in seen:
            out.append(dict(rec))
            seen.add(rec_key)
    return sorted(out, key=record_sort_key)


def _phase3_tie_beam_selection_pool(
    records: Sequence[Mapping[str, Any]],
    *,
    default_cap: int,
    score_key: str,
    score_ratio: float,
    abs_tol: float,
    max_branches: int,
    max_late_coordinate: float,
    min_depth_left: int,
    depth_one_based: int,
    max_depth_local: int,
    phase3_selector_score_key: str,
    phase3_record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    phase2_record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        [dict(rec) for rec in records],
        key=(
            phase3_record_sort_key
            if str(score_key) == str(phase3_selector_score_key)
            else phase2_record_sort_key
        ),
    )
    cap_default = int(max(1, default_cap))
    derived_depth_left = int(max(0, int(max_depth_local) - int(depth_one_based)))
    derived_late_coordinate = float(int(depth_one_based) / max(1, int(max_depth_local)))
    if not ordered:
        return [], {
            "active": False,
            "band_count": 0,
            "selected_count": 0,
            "best_score": float("-inf"),
            "depth_left": int(derived_depth_left),
            "late_coordinate": float(derived_late_coordinate),
            "reason": "empty",
        }
    best_score = float(ordered[0].get(score_key, float("-inf")))
    snapshot = _record_controller_snapshot(ordered[0])
    depth_left = int(
        max(
            0,
            int(snapshot.get("depth_left", derived_depth_left))
            if isinstance(snapshot, Mapping)
            else int(derived_depth_left),
        )
    )
    late_coordinate = float(
        snapshot.get("late_coordinate", derived_late_coordinate)
        if isinstance(snapshot, Mapping)
        else float(derived_late_coordinate)
    )
    criteria_enabled = bool(
        int(max_branches) > int(cap_default)
        and (
            (float(score_ratio) < 1.0 and math.isfinite(best_score) and best_score > 0.0)
            or (float(abs_tol) > 0.0 and math.isfinite(best_score))
        )
    )
    maturity_open = bool(
        int(depth_left) >= int(min_depth_left)
        and float(late_coordinate) <= float(max_late_coordinate)
    )
    if not criteria_enabled:
        return ordered[:cap_default], {
            "active": False,
            "band_count": int(min(len(ordered), cap_default)),
            "selected_count": int(min(len(ordered), cap_default)),
            "best_score": float(best_score),
            "depth_left": int(depth_left),
            "late_coordinate": float(late_coordinate),
            "reason": "disabled",
        }
    if not maturity_open:
        return ordered[:cap_default], {
            "active": False,
            "band_count": int(min(len(ordered), cap_default)),
            "selected_count": int(min(len(ordered), cap_default)),
            "best_score": float(best_score),
            "depth_left": int(depth_left),
            "late_coordinate": float(late_coordinate),
            "reason": "maturity_closed",
        }
    band: list[dict[str, Any]] = [dict(ordered[0])]
    seen = {_selection_record_key(ordered[0])}
    for rec in ordered[1:]:
        score_val = float(rec.get(score_key, float("-inf")))
        if not math.isfinite(score_val):
            continue
        within_ratio = bool(
            float(score_ratio) < 1.0 and score_val >= float(score_ratio) * best_score
        )
        within_abs = bool(
            float(abs_tol) > 0.0 and (best_score - score_val) <= float(abs_tol)
        )
        if not (within_ratio or within_abs):
            continue
        rec_key = _selection_record_key(rec)
        if rec_key in seen:
            continue
        band.append(dict(rec))
        seen.add(rec_key)
    if len(band) <= cap_default:
        return band[:cap_default], {
            "active": False,
            "band_count": int(len(band)),
            "selected_count": int(min(len(band), cap_default)),
            "best_score": float(best_score),
            "depth_left": int(depth_left),
            "late_coordinate": float(late_coordinate),
            "reason": "band_not_wider_than_default",
        }
    selected_cap = int(min(len(band), max(int(cap_default), int(max_branches))))
    return band[:selected_cap], {
        "active": True,
        "band_count": int(len(band)),
        "selected_count": int(selected_cap),
        "best_score": float(best_score),
        "depth_left": int(depth_left),
        "late_coordinate": float(late_coordinate),
        "reason": "phase3_score_band",
    }
