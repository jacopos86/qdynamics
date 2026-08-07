#!/usr/bin/env python3
"""Helpers for controller progress and partial-payload JSON emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.legacy.checkpoint_types import (
    high_miss_no_admit_diagnostic_counts,
    high_miss_no_admit_soft_fallback_counts,
)


"Built Math: progress = base(mode, counts, elapsed) ⊕ observable_metrics ⊕ extra."
def build_progress_payload(
    *,
    mode: str,
    append_count: int,
    prune_count: int,
    repair_count: int,
    repair_retry_attempt_count: int,
    trajectory_points: int,
    ledger_entries: int,
    logical_block_count: int,
    runtime_parameter_count: int,
    total_checkpoints: int,
    wallclock_elapsed_s: float | None,
    observable_metrics: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "running",
        "stage": "unknown",
        "mode": str(mode),
        "append_count": int(append_count),
        "prune_count": int(prune_count),
        "repair_count": int(repair_count),
        "repair_retry_attempt_count": int(repair_retry_attempt_count),
        "trajectory_points": int(trajectory_points),
        "ledger_entries": int(ledger_entries),
        "logical_block_count": int(logical_block_count),
        "runtime_parameter_count": int(runtime_parameter_count),
        "total_checkpoints": int(total_checkpoints),
        "wallclock_elapsed_s": (None if wallclock_elapsed_s is None else float(wallclock_elapsed_s)),
    }
    payload.update(dict(observable_metrics))
    payload.update(dict(extra))
    return payload


"Built Math: summary = Σ(ledger[action_kind]) with backend support set carried forward."
def build_partial_payload(
    *,
    status: str,
    stage: str,
    mode: str,
    trajectory: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    controller_state: Mapping[str, Any],
    logical_block_count: int,
    runtime_parameter_count: int,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executed_backends = sorted({str(row.get("decision_backend", "exact")) for row in ledger})
    payload_summary = (
        dict(summary)
        if summary is not None
        else {
            "append_count": int(sum(1 for row in ledger if str(row.get("action_kind")) == "append_candidate")),
            "prune_count": int(sum(1 for row in ledger if str(row.get("action_kind")) == "prune_coordinate")),
            "repair_count": int(sum(1 for row in ledger if str(row.get("action_kind", "")).startswith("repair_"))),
            "repair_retry_attempt_count": int(
                sum(
                    1
                    for row in ledger
                    if str(row.get("action_kind")) == "repair_miss"
                    and row.get("repair_max_attempts") is not None
                )
            ),
            "repair_retry_exhausted_count": int(
                sum(
                    1
                    for row in ledger
                    if str(row.get("repair_failure_reason", ""))
                    == "repair_retry_exhausted_high_miss_no_admit"
                )
            ),
            "stay_count": int(sum(1 for row in ledger if str(row.get("action_kind")) == "stay")),
            **high_miss_no_admit_soft_fallback_counts(ledger),
            **high_miss_no_admit_diagnostic_counts(ledger),
            "executed_decision_backends": list(executed_backends),
            "final_logical_block_count": int(logical_block_count),
            "final_runtime_parameter_count": int(runtime_parameter_count),
        }
    )
    return {
        "status": str(status),
        "stage": str(stage),
        "mode": str(mode),
        "trajectory": [dict(row) for row in trajectory],
        "ledger": [dict(row) for row in ledger],
        "reference": {
            "controller_state": dict(controller_state),
        },
        "summary": payload_summary,
    }


"Built Math: write(path, payload) = replace(path, json.dumps(payload))."
def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
