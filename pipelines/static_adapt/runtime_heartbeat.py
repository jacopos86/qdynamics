"""Stdlib-only live heartbeat helpers for static ADAPT subprocesses.

The helpers in this module are intentionally small and dependency-free so they
can be used from local runs, CHTC wrappers, and Optuna subprocess monitors
without importing the heavy ADAPT runtime.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_HEARTBEAT_SCHEMA = "static_adapt_live_heartbeat_v1"
_HEARTBEAT_EVENT_SCHEMA = "static_adapt_live_heartbeat_event_v1"
_AI_LOG_RE = re.compile(r"(?:^|\s)AI_LOG\s+(\{.*\})\s*$")

_PROGRESS_FIELDS = (
    "depth",
    "round",
    "current_depth",
    "ansatz_depth",
    "energy",
    "adapt_energy",
    "delta_e",
    "abs_delta_e",
    "delta_abs_current",
    "delta_abs_drop_from_prev",
    "gain",
    "max_grad",
    "drop_plateau_hits",
    "drop_patience",
    "adapt_drop_patience_resolved",
    "phase1_shortlist_size",
    "phase2_shortlist_size",
    "phase3_shortlist_size",
    "shortlist_size",
    "retained_shortlist_size",
    "raw_candidate_record_count",
    "phase2_raw_candidate_record_count",
    "admitted_record_count",
    "candidate_count",
    "available_count",
    "gradient_available_count",
    "frontier_input_count",
    "frontier_kept_count",
    "terminal_kept_count",
    "proposal_family_count",
    "proposals_selected_count",
    "live_branch_cap",
    "children_per_parent_cap",
    "terminated_keep_cap",
    "nfev_opt_so_far",
    "nfev_opt",
    "nit_opt",
    "best_fun",
    "delta_abs_best",
    "elapsed_opt_s",
    "run_elapsed_s",
    "gradient_eval_elapsed_s",
    "gradient_source",
    "beam_round_elapsed_s",
    "beam_parent_eval_elapsed_s",
    "beam_child_materialize_elapsed_s",
    "beam_parent_workers_requested",
    "beam_parent_workers_effective",
    "beam_parent_parallel_enabled",
    "beam_parent_parallel_disable_reasons",
    "beam_parent_parallel_merge_order",
    "optimizer_elapsed_s",
    "beam_enabled",
    "iter_elapsed_s",
    "opt_method",
    "iter",
    "stage",
    "selection_mode",
    "selected_op",
    "selected_logical_op",
    "stop_reason",
    "stop_reason_so_far",
)

_PROGRESS_ALIASES = {
    "ansatz_depth": "depth",
    "adapt_energy": "energy",
    "abs_delta_e": "delta_abs_current",
    "adapt_drop_patience_resolved": "drop_patience",
    "available_count": "gradient_available_count",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_ai_log_line(line: str) -> dict[str, Any] | None:
    """Return the JSON payload from an ``AI_LOG`` line, or ``None``.

    ADAPT emits lines as ``AI_LOG {json}``.  The parser also tolerates a small
    textual prefix before ``AI_LOG`` so external wrappers can tee annotated logs
    without breaking live status updates.
    """

    match = _AI_LOG_RE.search(str(line).strip())
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _progress_value(value: Any) -> Any:
    cleaned = _jsonable(value)
    if cleaned is None:
        return None
    return cleaned


def normalize_ai_log_progress(
    payload: Mapping[str, Any],
    *,
    elapsed_s: float,
    pid: int | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one ADAPT ``AI_LOG`` payload into a live heartbeat state."""

    previous_payload = dict(previous or {})
    progress = dict(previous_payload.get("progress", {}) if isinstance(previous_payload.get("progress"), Mapping) else {})
    for field in _PROGRESS_FIELDS:
        if field not in payload:
            continue
        value = _progress_value(payload.get(field))
        if value is None:
            continue
        progress[str(_PROGRESS_ALIASES.get(field, field))] = value

    event = payload.get("event")
    last_fields = {str(k): _jsonable(v) for k, v in payload.items() if str(k) != "event"}
    normalized = dict(previous_payload)
    normalized.update(
        {
            "schema": _HEARTBEAT_SCHEMA,
            "status": "running",
            "pid": None if pid is None else int(pid),
            "updated_utc": utc_now_iso(),
            "elapsed_s": float(elapsed_s),
            "returncode": None,
            "last_ai_log_event": None if event is None else str(event),
            "last_ai_log_ts_utc": utc_now_iso(),
            "last_ai_log_fields": last_fields,
            "progress": progress,
        }
    )
    if "errors" not in normalized or not isinstance(normalized.get("errors"), list):
        normalized["errors"] = []
    return normalized


class LiveHeartbeatRecorder:
    """Fail-soft writer for ``heartbeat.json`` plus optional JSONL events."""

    def __init__(
        self,
        *,
        heartbeat_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        event_jsonl_path: str | Path | None = None,
    ) -> None:
        self.heartbeat_path = Path(heartbeat_path)
        self.event_jsonl_path = None if event_jsonl_path in {None, ""} else Path(event_jsonl_path)  # type: ignore[arg-type]
        started = utc_now_iso()
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "schema": _HEARTBEAT_SCHEMA,
            "status": "created",
            "started_utc": started,
            "updated_utc": started,
            "elapsed_s": 0.0,
            "pid": None,
            "command": [],
            "returncode": None,
            "progress": {},
            "errors": [],
        }
        if metadata:
            self._state.update({str(k): _jsonable(v) for k, v in metadata.items()})

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def _persist_locked(self) -> None:
        try:
            _atomic_write_json(self.heartbeat_path, self._state)
        except Exception as exc:  # heartbeat writes must never fail the run
            self._state.setdefault("errors", []).append(f"heartbeat_write:{type(exc).__name__}:{exc}")

    def _append_event_locked(self, *, ai_log_payload: Mapping[str, Any] | None = None) -> None:
        if self.event_jsonl_path is None:
            return
        try:
            self.event_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "schema": _HEARTBEAT_EVENT_SCHEMA,
                "timestamp_utc": utc_now_iso(),
                "status": self._state.get("status"),
                "pid": self._state.get("pid"),
                "elapsed_s": self._state.get("elapsed_s"),
                "last_ai_log_event": self._state.get("last_ai_log_event"),
                "progress": dict(self._state.get("progress", {})),
            }
            if ai_log_payload is not None:
                event["ai_log"] = _jsonable(dict(ai_log_payload))
            with self.event_jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
        except Exception as exc:
            self._state.setdefault("errors", []).append(f"heartbeat_event_write:{type(exc).__name__}:{exc}")

    def mark_started(self, *, pid: int | None, command: Sequence[str]) -> dict[str, Any]:
        with self._lock:
            self._state.update(
                {
                    "status": "running",
                    "pid": None if pid is None else int(pid),
                    "command": [str(x) for x in command],
                    "updated_utc": utc_now_iso(),
                    "returncode": None,
                }
            )
            self._persist_locked()
            self._append_event_locked()
            return dict(self._state)

    def update_from_ai_log(
        self,
        payload: Mapping[str, Any],
        *,
        elapsed_s: float,
        pid: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            normalized = normalize_ai_log_progress(
                payload,
                elapsed_s=float(elapsed_s),
                pid=pid,
                previous=self._state,
            )
            self._state.update(normalized)
            self._persist_locked()
            self._append_event_locked(ai_log_payload=payload)
            return dict(self._state)

    def mark_finished(
        self,
        *,
        status: str,
        returncode: int | None,
        elapsed_s: float,
    ) -> dict[str, Any]:
        with self._lock:
            self._state.update(
                {
                    "status": str(status),
                    "returncode": None if returncode is None else int(returncode),
                    "elapsed_s": float(elapsed_s),
                    "updated_utc": utc_now_iso(),
                }
            )
            self._persist_locked()
            self._append_event_locked()
            return dict(self._state)
