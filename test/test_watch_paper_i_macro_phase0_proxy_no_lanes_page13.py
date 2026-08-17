from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import pytest

from pipelines.reporting import (
    watch_paper_i_macro_phase0_proxy_no_lanes_page13 as watcher,
)


EXECUTION_IDS = tuple(f"execution-{index}" for index in range(6))


def _previous_status(*, refreshed_count: int) -> dict[str, Any]:
    return {
        "schema": watcher.SCHEMA,
        "status": "watching_for_next_round50_completion",
        "refreshed_completed_execution_ids": list(
            EXECUTION_IDS[:refreshed_count]
        ),
        "page13_pdf": {"path": "fixture.pdf", "sha256": "pdf-sha"},
    }


def _serial_status(*, status: str, completed_count: int) -> dict[str, Any]:
    return {
        "status": status,
        "current_execution_id": (
            None
            if completed_count == len(EXECUTION_IDS)
            else EXECUTION_IDS[completed_count]
        ),
        "completed_execution_ids": list(EXECUTION_IDS[:completed_count]),
    }


def _patch_watch_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    previous: dict[str, Any] | None,
    serial_status: dict[str, Any],
) -> list[dict[str, Any]]:
    target_pdf = tmp_path / "page13-report.pdf"
    target_provenance = tmp_path / "page13-provenance.json"
    written: list[dict[str, Any]] = []

    monkeypatch.setattr(watcher, "expected_execution_ids", lambda: EXECUTION_IDS)
    monkeypatch.setattr(
        watcher,
        "_load_previous_status",
        lambda: copy.deepcopy(previous),
    )
    monkeypatch.setattr(
        watcher.page13,
        "load",
        lambda _path: copy.deepcopy(serial_status),
    )
    monkeypatch.setattr(
        watcher.page13,
        "verify_self_digest",
        lambda _value, *, label: None,
    )
    monkeypatch.setattr(
        watcher,
        "validated_completed_execution_ids",
        lambda value: tuple(value["completed_execution_ids"]),
    )
    monkeypatch.setattr(watcher.page13, "TARGET_PDF", target_pdf)
    monkeypatch.setattr(
        watcher.page13,
        "TARGET_PROVENANCE",
        target_provenance,
    )
    monkeypatch.setattr(watcher.page13, "sha256", lambda _path: "pdf-sha")
    monkeypatch.setattr(
        watcher.page13,
        "binding",
        lambda path: {
            "path": str(path),
            "sha256": (
                "pdf-sha" if path == target_pdf else "provenance-sha"
            ),
        },
    )
    monkeypatch.setattr(
        watcher,
        "_write_status",
        lambda value: written.append(copy.deepcopy(dict(value))),
    )
    return written


def _successful_refresh() -> dict[str, Any]:
    return {
        "status": "updated_existing_report_in_place",
        "page_count": 13,
        "pdf": {"path": "fixture.pdf", "sha256": "refreshed-pdf-sha"},
    }


def test_watch_does_not_refresh_an_unchanged_completed_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous = _previous_status(refreshed_count=2)
    serial_status = _serial_status(status="running", completed_count=2)
    written = _patch_watch_io(
        monkeypatch,
        tmp_path,
        previous=previous,
        serial_status=serial_status,
    )

    def unexpected_refresh() -> dict[str, Any]:
        raise AssertionError("unchanged completion prefix must not refresh")

    monkeypatch.setattr(watcher, "_run_updater", unexpected_refresh)

    assert watcher.watch(poll_seconds=1.0, once=True) == 0
    assert len(written) == 1
    assert written[0]["status"] == "watching_for_next_round50_completion"
    assert written[0]["refreshed_completed_execution_ids"] == list(
        EXECUTION_IDS[:2]
    )


def test_watch_refreshes_when_completed_prefix_grows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous = _previous_status(refreshed_count=1)
    serial_status = _serial_status(status="running", completed_count=2)
    written = _patch_watch_io(
        monkeypatch,
        tmp_path,
        previous=previous,
        serial_status=serial_status,
    )
    refresh_calls: list[None] = []

    def refresh() -> dict[str, Any]:
        refresh_calls.append(None)
        return _successful_refresh()

    monkeypatch.setattr(watcher, "_run_updater", refresh)

    assert watcher.watch(poll_seconds=1.0, once=True) == 0
    assert len(refresh_calls) == 1
    assert len(written) == 1
    assert written[0]["status"] == "watching_for_next_round50_completion"
    assert written[0]["refreshed_completed_execution_ids"] == list(
        EXECUTION_IDS[:2]
    )
    assert written[0]["last_refresh_result"] == _successful_refresh()


def test_watch_refreshes_sixth_cell_then_exits_terminally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous = _previous_status(refreshed_count=5)
    serial_status = _serial_status(status="passed", completed_count=6)
    written = _patch_watch_io(
        monkeypatch,
        tmp_path,
        previous=previous,
        serial_status=serial_status,
    )
    refresh_calls: list[None] = []

    def refresh() -> dict[str, Any]:
        refresh_calls.append(None)
        return _successful_refresh()

    monkeypatch.setattr(watcher, "_run_updater", refresh)

    assert watcher.watch(poll_seconds=1.0, once=False) == 0
    assert len(refresh_calls) == 1
    assert len(written) == 1
    assert written[0]["status"] == (
        "passed_all_six_round50_cells_refreshed"
    )
    assert written[0]["validated_completed_execution_ids"] == list(
        EXECUTION_IDS
    )
    assert written[0]["refreshed_completed_execution_ids"] == list(
        EXECUTION_IDS
    )


def test_watch_exits_nonzero_when_serial_run_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous = _previous_status(refreshed_count=2)
    serial_status = _serial_status(status="failed", completed_count=2)
    written = _patch_watch_io(
        monkeypatch,
        tmp_path,
        previous=previous,
        serial_status=serial_status,
    )

    def unexpected_refresh() -> dict[str, Any]:
        raise AssertionError("unchanged completion prefix must not refresh")

    monkeypatch.setattr(watcher, "_run_updater", unexpected_refresh)

    assert watcher.watch(poll_seconds=1.0, once=False) == 2
    assert len(written) == 1
    assert written[0]["status"] == "source_serial_run_failed"
    assert written[0]["serial_status"]["status"] == "failed"


def test_main_refuses_a_second_watcher_instance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(watcher, "LOCK_PATH", tmp_path / "watcher.lock")
    monkeypatch.setattr(sys, "argv", ["page13-watcher", "--once"])
    monkeypatch.setattr(
        watcher.fcntl,
        "flock",
        lambda _fileno, _operation: (_ for _ in ()).throw(BlockingIOError()),
    )
    monkeypatch.setattr(
        watcher,
        "watch",
        lambda **_kwargs: pytest.fail("second watcher must not enter watch()"),
    )

    assert watcher.main() == 3
    assert "already running" in capsys.readouterr().err
