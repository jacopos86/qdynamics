"""Controller phase-state accessors for static ADAPT.

This module owns pure reads from controller snapshots and candidate records. It
does not own controller updates, scoring, pruning, measurement work, or route
policy.
"""

from __future__ import annotations

from typing import Any, Mapping

from pipelines.scaffold.hh_continuation_types import CandidateFeatures

__all__ = [
    "_controller_cap",
    "_controller_phase_live",
    "_controller_phase_shots",
    "_controller_terminal_phase",
    "_controller_threshold",
    "_record_controller_snapshot",
]


def _record_controller_snapshot(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None
    snapshot_raw = record.get("controller_snapshot")
    if isinstance(snapshot_raw, Mapping):
        return dict(snapshot_raw)
    feat_obj = record.get("feature")
    snapshot_feat = (
        getattr(feat_obj, "controller_snapshot", None)
        if isinstance(feat_obj, CandidateFeatures)
        else None
    )
    if isinstance(snapshot_feat, Mapping):
        return dict(snapshot_feat)
    return None


def _controller_phase_live(
    snapshot: Any | None,
    phase_name: str,
    default_value: bool = True,
) -> bool:
    if snapshot is None:
        return bool(default_value)
    live_raw = getattr(snapshot, "phase_live", {})
    if isinstance(live_raw, Mapping) and str(phase_name) in live_raw:
        return bool(live_raw.get(str(phase_name)))
    if isinstance(snapshot, Mapping):
        live_mapping = snapshot.get("phase_live")
        if isinstance(live_mapping, Mapping) and str(phase_name) in live_mapping:
            return bool(live_mapping.get(str(phase_name)))
    return bool(default_value)


def _controller_phase_shots(
    snapshot: Any | None,
    phase_name: str,
    default_value: int = 1,
) -> int:
    if snapshot is None:
        return int(max(1, default_value))
    if not _controller_phase_live(snapshot, phase_name, True):
        return 0
    for attr_name in ("phase_shots_effective", "phase_shots"):
        values = getattr(snapshot, attr_name, {})
        if isinstance(snapshot, Mapping):
            values = snapshot.get(attr_name, values)
        if isinstance(values, Mapping) and str(phase_name) in values:
            try:
                return int(max(1, round(float(values.get(str(phase_name))))))
            except (TypeError, ValueError):
                pass
    return int(max(1, default_value))


def _controller_terminal_phase(
    snapshot: Any | None,
    default_value: int,
) -> int:
    raw: Any = None
    if snapshot is not None:
        raw = getattr(snapshot, "terminal_phase", None)
        if raw is None and isinstance(snapshot, Mapping):
            raw = snapshot.get("terminal_phase")
    try:
        value = int(raw if raw is not None else default_value)
    except (TypeError, ValueError):
        value = int(default_value)
    return int(min(3, max(1, value)))


def _controller_cap(snapshot: Any | None, phase_name: str, default_value: int) -> int:
    if snapshot is None:
        return int(max(1, default_value))
    if not _controller_phase_live(snapshot, phase_name, True):
        return 0
    caps = getattr(snapshot, "phase_caps", {})
    if isinstance(snapshot, Mapping):
        caps = snapshot.get("phase_caps", caps)
    return int(max(1, caps.get(str(phase_name), default_value) if isinstance(caps, Mapping) else default_value))


def _controller_threshold(snapshot: Any | None, phase_name: str) -> float:
    if snapshot is None:
        return 0.0
    thresholds = getattr(snapshot, "phase_thresholds", {})
    if isinstance(snapshot, Mapping):
        thresholds = snapshot.get("phase_thresholds", thresholds)
    return float(thresholds.get(str(phase_name), 0.0) if isinstance(thresholds, Mapping) else 0.0)
