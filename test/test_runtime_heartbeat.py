from __future__ import annotations

import json
from pathlib import Path

from pipelines.static_adapt.runtime_heartbeat import (
    LiveHeartbeatRecorder,
    normalize_ai_log_progress,
    parse_ai_log_line,
)


def test_parse_ai_log_line_and_normalize_progress() -> None:
    payload = parse_ai_log_line(
        'AI_LOG {"event":"hardcoded_adapt_iter_done","depth":3,"delta_abs_current":0.012,"available_count":17}'
    )

    assert payload is not None
    assert payload["event"] == "hardcoded_adapt_iter_done"
    assert parse_ai_log_line("not an AI log") is None
    assert parse_ai_log_line("AI_LOG not-json") is None

    normalized = normalize_ai_log_progress(payload, elapsed_s=4.5, pid=123)
    assert normalized["schema"] == "static_adapt_live_heartbeat_v1"
    assert normalized["pid"] == 123
    assert normalized["last_ai_log_event"] == "hardcoded_adapt_iter_done"
    assert normalized["progress"]["depth"] == 3
    assert normalized["progress"]["delta_abs_current"] == 0.012
    assert normalized["progress"]["gradient_available_count"] == 17


def test_live_heartbeat_recorder_writes_current_and_events(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat.json"
    events = tmp_path / "heartbeat_events.jsonl"
    recorder = LiveHeartbeatRecorder(
        heartbeat_path=heartbeat,
        event_jsonl_path=events,
        metadata={"record_id": "unit", "family": "hh"},
    )

    recorder.mark_started(pid=456, command=["python", "fake.py"])
    recorder.update_from_ai_log(
        {
            "event": "hardcoded_adapt_iter_done",
            "depth": 5,
            "energy": -1.25,
            "max_grad": 0.03,
            "drop_plateau_hits": 2,
            "adapt_drop_patience_resolved": 8,
            "phase1_shortlist_size": 13,
            "phase2_shortlist_size": 7,
            "phase3_shortlist_size": 3,
        },
        elapsed_s=1.25,
        pid=456,
    )
    recorder.mark_finished(status="completed", returncode=0, elapsed_s=2.0)

    data = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["record_id"] == "unit"
    assert data["returncode"] == 0
    assert data["last_ai_log_event"] == "hardcoded_adapt_iter_done"
    assert data["progress"]["depth"] == 5
    assert data["progress"]["drop_patience"] == 8
    assert data["progress"]["phase3_shortlist_size"] == 3

    event_rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert len(event_rows) >= 3
    assert any(row.get("last_ai_log_event") == "hardcoded_adapt_iter_done" for row in event_rows)
