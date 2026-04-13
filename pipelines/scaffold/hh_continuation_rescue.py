#!/usr/bin/env python3
"""Simulator-only overlap-guided rescue helpers for HH continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class RescueConfig:
    enabled: bool = False
    simulator_only: bool = True
    recent_drop_patience: int = 2
    weak_drop_threshold: float = 1e-6
    shortlist_flat_ratio: float = 0.95
    max_candidates: int = 6
    min_overlap_gain: float = 1e-7


def _cheap_score_value(rec: Mapping[str, Any]) -> float:
    raw = rec.get("cheap_score", rec.get("simple_score", float("-inf")))
    if raw is None:
        raw = rec.get("simple_score", float("-inf"))
    return float(raw)


def _full_or_cheap_score_value(rec: Mapping[str, Any]) -> float:
    raw = rec.get("full_v2_score", None)
    if raw is None:
        return _cheap_score_value(rec)
    return float(raw)


def _legacy_simple_tie_value(rec: Mapping[str, Any]) -> float:
    if str(rec.get("cheap_score_version", "")) == "phase3_cheap_ratio_v1":
        return 0.0
    raw = rec.get("simple_score", float("-inf"))
    if raw is None:
        raw = float("-inf")
    return float(raw)


def should_trigger_rescue(
    *,
    enabled: bool,
    exact_state_available: bool,
    residual_opened: bool,
    trough_detected: bool,
    history: Sequence[Mapping[str, Any]],
    shortlist_records: Sequence[Mapping[str, Any]],
    cfg: RescueConfig,
) -> tuple[bool, str]:
    if not bool(enabled):
        return False, "disabled"
    if not bool(exact_state_available):
        return False, "exact_state_unavailable"
    if not (bool(residual_opened) or bool(trough_detected)):
        return False, "residual_not_open_or_no_trough"
    need = int(max(1, cfg.recent_drop_patience))
    recent = [row for row in history if isinstance(row, Mapping)][-need:]
    if len(recent) < need:
        return False, "insufficient_history"
    if any(float(row.get("delta_abs_drop_from_prev", 1.0)) > float(cfg.weak_drop_threshold) for row in recent):
        return False, "drop_not_flat"
    if len(shortlist_records) < 2:
        return False, "shortlist_too_small"
    top = _full_or_cheap_score_value(shortlist_records[0])
    second = _full_or_cheap_score_value(shortlist_records[1])
    if top <= 0.0:
        return False, "nonpositive_shortlist"
    if second < float(cfg.shortlist_flat_ratio) * top:
        return False, "shortlist_not_flat"
    return True, "flat_drop_and_shortlist"


def rank_rescue_candidates(
    *,
    records: Sequence[Mapping[str, Any]],
    overlap_gain_fn: Callable[[Mapping[str, Any]], float],
    cfg: RescueConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for rec in list(records)[: int(max(1, cfg.max_candidates))]:
        gain = float(overlap_gain_fn(rec))
        ranked.append(
            {
                **dict(rec),
                "overlap_gain": float(gain),
            }
        )
    ranked = sorted(
        ranked,
        key=lambda rec: (
            -float(rec.get("overlap_gain", 0.0)),
            -_full_or_cheap_score_value(rec),
            -_cheap_score_value(rec),
            -_legacy_simple_tie_value(rec),
            int(rec.get("candidate_pool_index", -1)),
            int(rec.get("position_id", -1)),
        ),
    )
    if not ranked:
        return None, {"executed": False, "reason": "no_candidates", "ranked": []}
    best = dict(ranked[0])
    if float(best.get("overlap_gain", 0.0)) <= float(cfg.min_overlap_gain):
        return None, {
            "executed": True,
            "reason": "insufficient_overlap_gain",
            "ranked": [dict(x) for x in ranked],
        }
    return best, {
        "executed": True,
        "reason": "selected",
        "ranked": [dict(x) for x in ranked],
    }
