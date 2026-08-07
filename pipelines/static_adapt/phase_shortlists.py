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
    "_record_lane_shortlist_runtime",
    "_selection_pool_from_shortlist",
    "_selection_record_key",
    "lane_phase1_shortlist_records",
    "lane_phase2_health_shortlist_records",
    "lane_quota_pressure_budgets",
]


@dataclass(frozen=True)
class PhaseShortlistRuntime:
    phase2_score_cfg: FullScoreConfig
    feature_updater: Callable[[Any, Mapping[str, Any]], Any]
    lane_policy_active: bool
    lane_summary: MutableMapping[str, Any]
    phase1_lane_quota_pressure: float
    phase2_lane_quota_pressure: float
    phase2_lane_rel_threshold: float
    shortlist_lane_route: str
    shortlist_lane_key: str
    shortlist_lanes: tuple[str, ...]
    shortlist_fallback_lane: str
    shortlist_lane_health_key_prefix: str
    physical_operator_identity_caps_enabled: bool = True
    phase1_lane_retention_enabled: bool = True


def _shortlist_lane_policy_active(runtime: PhaseShortlistRuntime) -> bool:
    return bool(runtime.lane_policy_active)


def _physical_operator_identity_caps_active(runtime: PhaseShortlistRuntime) -> bool:
    return bool(
        str(runtime.shortlist_lane_route) == "physical_operator_type"
        and runtime.physical_operator_identity_caps_enabled
    )


def _post_split_candidate_population(
    records: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether child projection has already changed representation.

    Physical-family protection is a parent Phase-I/II policy.  Once a
    single-Pauli child or child-set record exists, the population must be
    ranked globally; inherited/fallback parent lane labels have no authority.
    """

    for record in records:
        feature = record.get("feature")
        feature_payload = (
            dict(feature.__dict__)
            if isinstance(feature, CandidateFeatures)
            else (dict(feature) if isinstance(feature, Mapping) else {})
        )
        mode = str(
            record.get(
                "runtime_split_mode",
                feature_payload.get("runtime_split_mode", "off"),
            )
        ).strip().lower()
        representation = str(
            record.get(
                "runtime_split_chosen_representation",
                feature_payload.get(
                    "runtime_split_chosen_representation",
                    "parent",
                ),
            )
        ).strip().lower()
        if mode not in {"", "off", "none", "false", "0", "disabled"}:
            return True
        if representation not in {"", "parent"}:
            return True
    return False


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


def _global_parent_identity_representative_shortlist(
    records: Sequence[Mapping[str, Any]],
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    cap: int,
    tie_break_score_key: str | None,
    shortlist_flag: str | None,
) -> list[dict[str, Any]]:
    """Globally cap post-split records by their retained parent identity.

    Runtime-split children inherit a stable macro/pool identity from the
    retained parent that exposed them.  Phase II keeps only the best-scoring
    child representative for each such parent before applying its global cap.
    This is a lineage/identity decision: no physical-family lane field,
    budget, health threshold, or fallback lane is consulted.
    """

    population = _macro_identity_selection_population(
        records,
        score_key=score_key,
        tie_break_score_key=tie_break_score_key,
    )
    return _phase_shortlist_with_legacy_hook(
        list(population.representatives),
        runtime=runtime,
        score_key=score_key,
        threshold=float("-inf"),
        cap=cap,
        frontier_ratio=0.0,
        tie_break_score_key=tie_break_score_key,
        shortlist_flag=shortlist_flag,
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
    # Identity expansion is the final immutable candidate-position
    # population seen by the typed controller.  Freeze a deterministic rank
    # on that expanded population rather than retaining the representative's
    # identity-level rank (which can be duplicated across insertion
    # positions).
    expanded_size = int(len(expanded))
    for rank, row in enumerate(expanded, start=1):
        row["shortlist_rank"] = int(rank)
        row["shortlist_size"] = expanded_size
        feature = row.get("feature")
        if feature is not None:
            row["feature"] = runtime.feature_updater(
                feature,
                {
                    "shortlist_rank": int(rank),
                    "shortlist_size": expanded_size,
                },
            )
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
        "positions_considered": sorted(
            {
                int(rec.get("position_id", append_position_value))
                for rec in records_list
            }
        ),
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


def _finite_record_score(
    record: Mapping[str, Any],
    key: str | None,
    default: float = float("-inf"),
) -> float:
    if key is None:
        return 0.0
    raw = record.get(str(key), default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _normalized_lanes(lanes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(lane) for lane in lanes)
    if not normalized:
        raise ValueError("Physical-family shortlist lanes must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Physical-family shortlist lanes must be unique and ordered."
        )
    return normalized


def _record_lane(
    record: Mapping[str, Any],
    lane_key: str,
    *,
    lanes: Sequence[str],
    fallback_lane: str,
) -> str:
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    if fallback not in lane_choices:
        raise ValueError(
            f"Fallback lane {fallback!r} is absent from {lane_choices!r}."
        )
    lane = str(record.get(str(lane_key), fallback))
    return lane if lane in lane_choices else fallback


def _record_rank_key(
    record: Mapping[str, Any],
    *,
    score_key: str,
    tie_break_score_key: str | None = None,
) -> tuple[float, float, int, int]:
    return (
        -_finite_record_score(record, score_key),
        -_finite_record_score(record, tie_break_score_key),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _lane_budget_allocation(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    cap: int,
    lane_budgets: Mapping[str, int] | None,
    lanes: Sequence[str],
) -> dict[str, int]:
    cap_eff = int(max(0, cap))
    lane_choices = _normalized_lanes(lanes)
    budgets = {lane: 0 for lane in lane_choices}
    if cap_eff <= 0:
        return budgets
    if lane_budgets is not None:
        for lane in lane_choices:
            available = len(grouped.get(lane, ()))
            if available > 0:
                budgets[lane] = int(
                    min(available, max(0, lane_budgets.get(lane, 0)))
                )
        total = sum(budgets.values())
        if total > cap_eff:
            remaining = cap_eff
            for lane in lane_choices:
                keep = min(int(budgets[lane]), int(remaining))
                budgets[lane] = int(keep)
                remaining -= int(keep)
        return budgets

    nonempty = [
        lane for lane in lane_choices if len(grouped.get(lane, ())) > 0
    ]
    remaining = cap_eff
    for lane in nonempty:
        if remaining <= 0:
            break
        budgets[lane] = 1
        remaining -= 1
    while remaining > 0 and nonempty:
        progressed = False
        for lane in sorted(
            nonempty,
            key=lambda key: (
                -len(grouped.get(key, ())),
                lane_choices.index(key),
            ),
        ):
            if (
                budgets[lane] >= len(grouped.get(lane, ()))
                or remaining <= 0
            ):
                continue
            budgets[lane] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
    return budgets


def lane_quota_pressure_budgets(
    records: Sequence[Mapping[str, Any]],
    *,
    cap: int,
    score_key: str,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    threshold: float = float("-inf"),
    pressure: float = 1.0,
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    tie_break_score_key: str | None = None,
) -> dict[str, int]:
    """Allocate deterministic budgets across physical operator families."""

    cap_eff = int(max(0, cap))
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    if fallback not in lane_choices:
        raise ValueError(
            f"Fallback lane {fallback!r} is absent from {lane_choices!r}."
        )
    budgets = {lane: 0 for lane in lane_choices}
    if cap_eff <= 0:
        return budgets
    eligible = [
        dict(record)
        for record in records
        if _finite_record_score(record, score_key) >= float(threshold)
    ]
    if not eligible:
        eligible = [dict(record) for record in records]
    if not eligible:
        return budgets

    grouped: dict[str, list[dict[str, Any]]] = {
        lane: [] for lane in lane_choices
    }
    for record in eligible:
        grouped[
            _record_lane(
                record,
                lane_key,
                lanes=lane_choices,
                fallback_lane=fallback,
            )
        ].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )

    lane_health = {
        lane: (
            _finite_record_score(
                grouped[lane][0],
                score_key,
                default=0.0,
            )
            if grouped[lane]
            else 0.0
        )
        for lane in lane_choices
    }
    health_star = max(lane_health.values()) if lane_health else 0.0
    abs_floor = float(
        threshold if lane_abs_threshold is None else lane_abs_threshold
    )
    rel_floor = float(max(0.0, min(1.0, lane_rel_threshold)))
    live_lanes: list[str] = []
    for lane in lane_choices:
        if not grouped[lane]:
            continue
        relative = (
            float(lane_health[lane] / health_star)
            if health_star > 0.0
            else 0.0
        )
        if lane_health[lane] >= abs_floor and relative >= rel_floor:
            live_lanes.append(lane)
    if not live_lanes:
        return budgets

    live_records = [
        row for lane in live_lanes for row in grouped[lane]
    ]
    target = int(min(cap_eff, len(live_records)))
    if target <= 0:
        return budgets

    ranked_lanes = sorted(
        live_lanes,
        key=lambda lane: _record_rank_key(
            grouped[lane][0],
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )
    pressure_val = float(max(0.0, min(1.0, pressure)))
    reserved_lane_count = int(math.ceil(pressure_val * len(ranked_lanes)))
    reserved_lane_count = int(
        min(target, len(ranked_lanes), max(0, reserved_lane_count))
    )
    for lane in ranked_lanes[:reserved_lane_count]:
        budgets[lane] = 1

    remaining = int(target - sum(budgets.values()))
    globally_ranked = sorted(
        (
            record
            for lane in live_lanes
            for record in grouped[lane][int(budgets.get(lane, 0)) :]
        ),
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )
    for record in globally_ranked:
        if remaining <= 0:
            break
        lane = _record_lane(
            record,
            lane_key,
            lanes=lane_choices,
            fallback_lane=fallback,
        )
        if lane not in live_lanes or budgets[lane] >= len(grouped[lane]):
            continue
        budgets[lane] += 1
        remaining -= 1
    return budgets


def _record_identity(
    record: Mapping[str, Any],
) -> tuple[str, str, int, int]:
    return (
        str(record.get("candidate_label", "")),
        str(record.get("generator_id", "")),
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
    )


def _global_refill_shortlist(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    cap: int,
    tie_break_score_key: str | None = None,
) -> list[dict[str, Any]]:
    cap_eff = int(max(0, cap))
    out = [dict(row) for row in selected]
    if cap_eff <= 0:
        return []
    target = int(min(cap_eff, len(candidates)))
    if len(out) >= target:
        return out[:target]
    seen = {_record_identity(row) for row in out}
    for row in sorted(
        [dict(candidate) for candidate in candidates],
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    ):
        key = _record_identity(row)
        if key in seen:
            continue
        out.append(dict(row))
        seen.add(key)
        if len(out) >= target:
            break
    return out


def _frontier_take(
    ranked: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    cap: int,
    frontier_ratio: float,
    score_eps: float = 1.0e-12,
) -> list[dict[str, Any]]:
    if int(cap) <= 0 or not ranked:
        return []
    cap_eff = int(max(1, min(int(cap), len(ranked))))
    take = int(cap_eff)
    frontier_cut = float(max(0.0, min(1.0, frontier_ratio)))
    if cap_eff > 1 and 0.0 < frontier_cut < 1.0:
        for idx in range(cap_eff - 1):
            current = _finite_record_score(
                ranked[idx],
                score_key,
                default=0.0,
            )
            following = _finite_record_score(
                ranked[idx + 1],
                score_key,
                default=0.0,
            )
            ratio = float(
                (following + float(score_eps))
                / (current + float(score_eps))
            )
            if ratio <= frontier_cut:
                take = int(idx + 1)
                break
    return [dict(row) for row in ranked[:take]]


def _mark_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    shortlist_flag: str | None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    size = int(len(records))
    for rank, record in enumerate(records, start=1):
        updated = dict(record)
        updated["shortlist_rank"] = int(rank)
        updated["shortlist_size"] = int(size)
        if shortlist_flag is not None:
            updated[str(shortlist_flag)] = True
        feature = updated.get("feature")
        if feature_updater is not None and feature is not None:
            payload: dict[str, Any] = {
                "shortlist_rank": int(rank),
                "shortlist_size": int(size),
            }
            if shortlist_flag is not None:
                payload[str(shortlist_flag)] = True
            updated["feature"] = feature_updater(feature, payload)
        out.append(updated)
    return out


def lane_phase1_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Select Phase-I records while retaining physical-family coverage."""

    cap_eff = int(max(0, cap))
    if cap_eff <= 0:
        return []
    filtered = [
        dict(record)
        for record in records
        if _finite_record_score(record, score_key) >= float(threshold)
    ]
    if not filtered:
        filtered = [dict(record) for record in records]
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    grouped: dict[str, list[dict[str, Any]]] = {
        lane: [] for lane in lane_choices
    }
    for record in filtered:
        grouped[
            _record_lane(
                record,
                lane_key,
                lanes=lane_choices,
                fallback_lane=fallback,
            )
        ].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
    budgets = _lane_budget_allocation(
        grouped,
        cap=cap_eff,
        lane_budgets=lane_budgets,
        lanes=lane_choices,
    )
    selected: list[dict[str, Any]] = []
    for lane in lane_choices:
        selected.extend(
            _frontier_take(
                grouped[lane],
                score_key=score_key,
                cap=int(budgets.get(lane, 0)),
                frontier_ratio=float(frontier_ratio),
            )
        )
    selected = _global_refill_shortlist(
        selected,
        filtered,
        score_key=score_key,
        cap=cap_eff,
        tie_break_score_key=tie_break_score_key,
    )
    if not selected and filtered:
        selected = [
            sorted(
                filtered,
                key=lambda record: _record_rank_key(
                    record,
                    score_key=score_key,
                    tie_break_score_key=tie_break_score_key,
                ),
            )[0]
        ]
    selected = sorted(
        selected,
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )[:cap_eff]
    return _mark_shortlist_records(
        selected,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
    )


def lane_phase2_health_shortlist_records(
    records: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    lane_key: str,
    lanes: Sequence[str],
    fallback_lane: str,
    lane_abs_threshold: float | None = None,
    lane_rel_threshold: float = 0.0,
    lane_budgets: Mapping[str, int] | None = None,
    tie_break_score_key: str | None = None,
    shortlist_flag: str | None = None,
    feature_updater: Callable[[Any, Mapping[str, Any]], Any] | None = None,
    health_key_prefix: str = "physical_operator",
) -> list[dict[str, Any]]:
    """Select Phase-II records from geometry-healthy physical families."""

    cap_eff = int(max(0, cap))
    if cap_eff <= 0:
        return []
    lane_choices = _normalized_lanes(lanes)
    fallback = str(fallback_lane)
    health_prefix = str(health_key_prefix)
    lane_health_key = f"{health_prefix}_lane_health"
    lane_relative_health_key = f"{health_prefix}_lane_relative_health"
    lane_live_key = f"{health_prefix}_lane_live"
    grouped: dict[str, list[dict[str, Any]]] = {
        lane: [] for lane in lane_choices
    }
    for record in records:
        grouped[
            _record_lane(
                record,
                lane_key,
                lanes=lane_choices,
                fallback_lane=fallback,
            )
        ].append(dict(record))
    for lane in lane_choices:
        grouped[lane] = sorted(
            grouped[lane],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )
    lane_health = {
        lane: (
            _finite_record_score(
                grouped[lane][0],
                score_key,
                default=0.0,
            )
            if grouped[lane]
            else 0.0
        )
        for lane in lane_choices
    }
    health_star = max(lane_health.values()) if lane_health else 0.0
    abs_floor = float(
        threshold if lane_abs_threshold is None else lane_abs_threshold
    )
    rel_floor = float(max(0.0, min(1.0, lane_rel_threshold)))
    live_lanes: list[str] = []
    for lane in lane_choices:
        relative = (
            float(lane_health[lane] / health_star)
            if health_star > 0.0
            else 0.0
        )
        live = bool(
            grouped[lane]
            and lane_health[lane] >= abs_floor
            and relative >= rel_floor
        )
        if live:
            live_lanes.append(lane)
        for idx, row in enumerate(grouped[lane]):
            grouped[lane][idx] = {
                **dict(row),
                lane_health_key: float(lane_health[lane]),
                lane_relative_health_key: float(relative),
                lane_live_key: bool(live),
            }
    budgets = _lane_budget_allocation(
        {lane: grouped[lane] for lane in live_lanes},
        cap=cap_eff,
        lane_budgets=lane_budgets,
        lanes=lane_choices,
    )
    selected: list[dict[str, Any]] = []
    live_candidates: list[dict[str, Any]] = []
    for lane in lane_choices:
        if lane not in live_lanes:
            continue
        candidates = [
            row
            for row in grouped[lane]
            if _finite_record_score(row, score_key) >= float(threshold)
        ]
        live_candidates.extend(dict(row) for row in candidates)
        selected.extend(
            _frontier_take(
                candidates,
                score_key=score_key,
                cap=int(budgets.get(lane, 0)),
                frontier_ratio=float(frontier_ratio),
            )
        )
    selected = _global_refill_shortlist(
        selected,
        live_candidates,
        score_key=score_key,
        cap=cap_eff,
        tie_break_score_key=tie_break_score_key,
    )
    if not selected:
        selected = sorted(
            [dict(row) for row in records],
            key=lambda record: _record_rank_key(
                record,
                score_key=score_key,
                tie_break_score_key=tie_break_score_key,
            ),
        )[:1]
    selected = sorted(
        selected,
        key=lambda record: _record_rank_key(
            record,
            score_key=score_key,
            tie_break_score_key=tie_break_score_key,
        ),
    )[:cap_eff]
    marked: list[dict[str, Any]] = []
    for row in selected:
        updates = {
            lane_health_key: row.get(lane_health_key),
            lane_relative_health_key: row.get(lane_relative_health_key),
            lane_live_key: row.get(lane_live_key),
        }
        feature = row.get("feature")
        updated = dict(row)
        if feature_updater is not None and feature is not None:
            updated["feature"] = feature_updater(feature, updates)
        marked.append(updated)
    return _mark_shortlist_records(
        marked,
        shortlist_flag=shortlist_flag,
        feature_updater=feature_updater,
    )


def _record_lane_shortlist_runtime(
    phase: str,
    *,
    runtime: PhaseShortlistRuntime,
    lane_budgets: Mapping[str, int],
    shortlist_size: int,
) -> None:
    summary = runtime.lane_summary
    runtime_payload = summary.setdefault("shortlist_runtime", {})
    normalized = {
        str(lane): int(max(0, lane_budgets.get(str(lane), 0)))
        for lane in runtime.shortlist_lanes
    }
    runtime_payload["lane_route"] = str(runtime.shortlist_lane_route)
    runtime_payload["lane_key"] = str(runtime.shortlist_lane_key)
    runtime_payload["lanes"] = [
        str(lane) for lane in runtime.shortlist_lanes
    ]
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
    summary = runtime.lane_summary
    runtime_payload = summary.setdefault("shortlist_runtime", {})
    runtime_payload[f"{phase}_shortlist_unit"] = ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR
    runtime_payload[f"{phase}_input_identity_count"] = int(input_identity_count)
    runtime_payload[f"{phase}_selected_identity_count"] = int(selected_identity_count)
    runtime_payload[f"{phase}_expanded_position_record_count"] = int(
        expanded_record_count
    )


def _phase1_selection_key(
    record: Mapping[str, Any],
    *,
    identity_selection_active: bool,
) -> str:
    if identity_selection_active:
        return str(record.get("route_a_shortlist_identity", ""))
    return (
        f"{macro_operator_identity(record)}"
        f"@position:{int(record.get('position_id', -1))}"
    )


def _compact_phase1_retention_record(
    record: Mapping[str, Any],
    *,
    score_key: str,
    identity_selection_active: bool,
) -> dict[str, Any]:
    feature = record.get("feature")
    feature_payload = (
        dict(feature.__dict__)
        if isinstance(feature, CandidateFeatures)
        else (dict(feature) if isinstance(feature, Mapping) else {})
    )
    snapshot_raw = record.get("controller_snapshot")
    if not isinstance(snapshot_raw, Mapping):
        snapshot_raw = feature_payload.get("controller_snapshot")
    snapshot = dict(snapshot_raw) if isinstance(snapshot_raw, Mapping) else {}
    return {
        "selection_key": _phase1_selection_key(
            record,
            identity_selection_active=identity_selection_active,
        ),
        "macro_operator_identity": str(macro_operator_identity(record)),
        "generator_id": (
            None
            if record.get("generator_id", feature_payload.get("generator_id")) is None
            else str(record.get("generator_id", feature_payload.get("generator_id")))
        ),
        "parent_generator_id": (
            None
            if record.get(
                "parent_generator_id",
                feature_payload.get("parent_generator_id"),
            )
            is None
            else str(
                record.get(
                    "parent_generator_id",
                    feature_payload.get("parent_generator_id"),
                )
            )
        ),
        "candidate_label": str(
            record.get(
                "candidate_label",
                feature_payload.get("candidate_label", ""),
            )
        ),
        "candidate_pool_index": int(record.get("candidate_pool_index", -1)),
        "position_id": int(record.get("position_id", -1)),
        "physical_operator_lane": (
            None
            if record.get(
                "physical_operator_lane",
                feature_payload.get("physical_operator_lane"),
            )
            is None
            else str(
                record.get(
                    "physical_operator_lane",
                    feature_payload.get("physical_operator_lane"),
                )
            )
        ),
        "score": float(record.get(score_key, float("-inf"))),
        "controller_step_index": (
            None
            if snapshot.get("step_index") is None
            else int(snapshot["step_index"])
        ),
    }


def _record_phase1_lane_retention_audit(
    *,
    runtime: PhaseShortlistRuntime,
    score_key: str,
    threshold: float,
    cap: int,
    frontier_ratio: float,
    tie_break_score_key: str | None,
    identity_selection_active: bool,
    global_shortlisted: Sequence[Mapping[str, Any]],
    lane_protected_shortlisted: Sequence[Mapping[str, Any]],
    lane_budgets: Mapping[str, int],
) -> None:
    summary = runtime.lane_summary
    global_rows = [
        _compact_phase1_retention_record(
            record,
            score_key=score_key,
            identity_selection_active=identity_selection_active,
        )
        for record in global_shortlisted
    ]
    lane_rows = [
        _compact_phase1_retention_record(
            record,
            score_key=score_key,
            identity_selection_active=identity_selection_active,
        )
        for record in lane_protected_shortlisted
    ]
    global_keys = [str(row["selection_key"]) for row in global_rows]
    lane_keys = [str(row["selection_key"]) for row in lane_rows]
    global_key_set = set(global_keys)
    lane_key_set = set(lane_keys)
    global_macro_ids = {
        str(row["macro_operator_identity"]) for row in global_rows
    }
    lane_macro_ids = {
        str(row["macro_operator_identity"]) for row in lane_rows
    }
    audits = summary.setdefault("phase1_lane_retention_audits", [])
    audits.append(
        {
            "schema": "paper_i_phase1_lane_retention_ablation_audit_v1",
            "audit_index": int(len(audits)),
            "classification_route": str(runtime.shortlist_lane_route),
            "classification_active": bool(_shortlist_lane_policy_active(runtime)),
            "lane_retention_enabled": bool(
                runtime.phase1_lane_retention_enabled
            ),
            "applied_selection_mode": "global",
            "selection_unit": (
                ROUTE_A_SHORTLIST_UNIT_MACRO_OPERATOR
                if identity_selection_active
                else "candidate_position_record"
            ),
            "score_key": str(score_key),
            "threshold": float(threshold),
            "cap": int(cap),
            "frontier_ratio": float(frontier_ratio),
            "tie_break_score_key": (
                None
                if tie_break_score_key is None
                else str(tie_break_score_key)
            ),
            "global_shortlist": global_rows,
            "lane_protected_counterfactual_shortlist": lane_rows,
            "lane_protected_counterfactual_lane_budgets": {
                str(lane): int(max(0, lane_budgets.get(str(lane), 0)))
                for lane in runtime.shortlist_lanes
            },
            "shortlists_differ": bool(global_keys != lane_keys),
            "lane_protected_only_selection_keys": [
                key for key in lane_keys if key not in global_key_set
            ],
            "global_only_selection_keys": [
                key for key in global_keys if key not in lane_key_set
            ],
            "lane_protected_only_macro_operator_identities": sorted(
                lane_macro_ids - global_macro_ids
            ),
            "global_only_macro_operator_identities": sorted(
                global_macro_ids - lane_macro_ids
            ),
        }
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
    if _post_split_candidate_population(records_list):
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
        pressure=float(runtime.phase1_lane_quota_pressure),
        lane_key=str(runtime.shortlist_lane_key),
        lanes=tuple(runtime.shortlist_lanes),
        fallback_lane=str(runtime.shortlist_fallback_lane),
        tie_break_score_key=tie_break_score_key,
    )
    lane_protected_shortlisted = lane_phase1_shortlist_records(
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
            if (
                identity_population_value is not None
                or not runtime.phase1_lane_retention_enabled
            )
            else runtime.feature_updater
        ),
    )
    if runtime.phase1_lane_retention_enabled:
        shortlisted = lane_protected_shortlisted
        actual_lane_budgets = lane_budgets
    else:
        shortlisted = phase_shortlist_records(
            selection_records,
            score_key=score_key,
            threshold=selection_threshold,
            cap=cap,
            frontier_ratio=selection_frontier_ratio,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )
        if not shortlisted and selection_records:
            shortlisted = phase_shortlist_records(
                selection_records,
                score_key=score_key,
                threshold=float("-inf"),
                cap=1,
                frontier_ratio=0.0,
                tie_break_score_key=tie_break_score_key,
                shortlist_flag=shortlist_flag,
            )
        _record_phase1_lane_retention_audit(
            runtime=runtime,
            score_key=score_key,
            threshold=selection_threshold,
            cap=cap,
            frontier_ratio=selection_frontier_ratio,
            tie_break_score_key=tie_break_score_key,
            identity_selection_active=bool(identity_population_value is not None),
            global_shortlisted=shortlisted,
            lane_protected_shortlisted=lane_protected_shortlisted,
            lane_budgets=lane_budgets,
        )
        actual_lane_budgets = {}
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
        _record_lane_shortlist_runtime(
            "phase1",
            runtime=runtime,
            lane_budgets=actual_lane_budgets,
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
        pressure=float(runtime.phase1_lane_quota_pressure),
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
    _record_lane_shortlist_runtime(
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
    if _post_split_candidate_population(records_list):
        return _global_parent_identity_representative_shortlist(
            records_list,
            runtime=runtime,
            score_key=score_key,
            cap=cap,
            tie_break_score_key=tie_break_score_key,
            shortlist_flag=shortlist_flag,
        )
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
    lane_relative_threshold = float(runtime.phase2_lane_rel_threshold)
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
        pressure=float(runtime.phase2_lane_quota_pressure),
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
        _record_lane_shortlist_runtime(
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
        pressure=float(runtime.phase2_lane_quota_pressure),
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
    _record_lane_shortlist_runtime(
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
