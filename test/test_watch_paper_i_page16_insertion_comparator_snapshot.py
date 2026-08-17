from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.reporting import (
    watch_paper_i_page16_insertion_comparator_snapshot as watcher,
)


SHA = "a" * 64


def _campaign(
    *,
    closed: tuple[str, ...] = (),
    unclosed: tuple[str, ...] = (),
    terminal_failed: bool = False,
    all_closed: bool = False,
    revisions: dict[str, str] | None = None,
    all_continuations_closed: bool = True,
    macro_terminal_authenticated: bool = True,
    eligible_continuations: tuple[str, ...] = (),
    continuation_revision: str = "c" * 64,
) -> dict[str, object]:
    expected = tuple(dict.fromkeys((*closed, *unclosed, "pending-id")))
    evidence_revisions = (
        {execution_id: "a" * 64 for execution_id in closed}
        if revisions is None
        else dict(revisions)
    )
    return {
        "inventory": {
            "campaign_state": "runtime_materialized",
            "execution_ids": list(expected),
            "evidence_revisions": evidence_revisions,
            "all_required_continuations_closed": all_continuations_closed,
            "macro_terminal_authenticated": macro_terminal_authenticated,
            "continuation_terminal_authority_satisfied": (
                all_continuations_closed
                and macro_terminal_authenticated
            ),
            "continuation_evidence_revision": continuation_revision,
            "continuation": {
                "eligible_execution_ids": list(eligible_continuations),
            },
        },
        "closed_execution_ids": closed,
        "unclosed_published_execution_ids": unclosed,
        "wave_statuses": {},
        "supervisor_status": None,
        "active_campaign_processes": [],
        "explicit_failures": (
            [{"source": "wave_1", "status": "failed"}]
            if terminal_failed
            else []
        ),
        "terminal_failed_no_active_supervisor_path": terminal_failed,
        "expected_local_execution_ids": closed if all_closed else expected,
        "all_ten_local_cells_closed": False,
        "all_nine_local_cells_closed": all_closed,
        "all_required_continuations_closed": all_continuations_closed,
        "macro_terminal_authenticated": macro_terminal_authenticated,
        "continuation_evidence_revision": continuation_revision,
        "continuation_terminal_authority_satisfied": (
            all_continuations_closed and macro_terminal_authenticated
        ),
    }


def _patch_watch_io(
    monkeypatch: pytest.MonkeyPatch,
    campaign: dict[str, object],
    *,
    reported: tuple[str, ...] = (),
    reported_revisions: dict[str, str] | None = None,
    reported_continuation_revision: str = "c" * 64,
) -> list[dict[str, object]]:
    written: list[dict[str, object]] = []
    monkeypatch.setattr(watcher, "_load_previous_status", lambda: None)
    monkeypatch.setattr(watcher, "_campaign_snapshot", lambda **_kwargs: campaign)
    monkeypatch.setattr(watcher, "_reported_local_execution_ids", lambda: reported)
    monkeypatch.setattr(
        watcher,
        "_reported_local_evidence_revisions",
        lambda: (
            {execution_id: "a" * 64 for execution_id in reported}
            if reported_revisions is None
            else dict(reported_revisions)
        ),
    )
    monkeypatch.setattr(watcher, "_candidate_chtc_execution_ids", lambda: ())
    monkeypatch.setattr(watcher, "_reported_chtc_execution_ids", lambda: ())
    monkeypatch.setattr(
        watcher,
        "_reported_continuation_evidence_revision",
        lambda: reported_continuation_revision,
    )

    def write_status(value: dict[str, object]) -> dict[str, object]:
        written.append(value)
        return value

    monkeypatch.setattr(watcher, "_write_status", write_status)
    return written


def test_closed_local_cell_triggers_one_page17_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution_id = "closed-id"
    campaign = _campaign(closed=(execution_id,))
    written = _patch_watch_io(monkeypatch, campaign)
    calls: list[dict[str, object]] = []

    def run_updater(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "updated_existing_report_in_place",
            "page_count": 17,
            "preserved_page_count": 16,
            "pdf": {"sha256": SHA, "size_bytes": 10},
            "reported_local_execution_ids": [execution_id],
            "reported_local_evidence_revisions": {execution_id: "a" * 64},
            "reported_continuation_evidence_revision": "c" * 64,
            "reported_chtc_execution_ids": [],
        }

    monkeypatch.setattr(watcher, "_run_updater", run_updater)
    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert len(calls) == 1
    assert written[-1]["status"] == "watching_for_next_authenticated_local_closure"
    assert written[-1]["refreshed_local_execution_ids"] == [execution_id]
    assert written[-1]["paper_evidence_adopted"] is False


def test_unclosed_publication_is_status_only_and_does_not_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    campaign = _campaign(unclosed=("unclosed-id",))
    written = _patch_watch_io(monkeypatch, campaign)
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("unclosed output must not refresh Page 17"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert written[-1]["unclosed_published_execution_ids"] == ["unclosed-id"]
    assert written[-1]["reported_local_execution_ids"] == []


def test_new_chtc_closure_receipt_triggers_page17_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    campaign = _campaign()
    written = _patch_watch_io(monkeypatch, campaign)
    remote_id = "strong-weak-always"
    monkeypatch.setattr(
        watcher, "_candidate_chtc_execution_ids", lambda: (remote_id,)
    )
    calls: list[dict[str, object]] = []

    def run_updater(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "updated_existing_report_in_place",
            "page_count": 17,
            "preserved_page_count": 16,
            "pdf": {"sha256": SHA, "size_bytes": 10},
            "reported_local_execution_ids": [],
            "reported_local_evidence_revisions": {},
            "reported_continuation_evidence_revision": "c" * 64,
            "reported_chtc_execution_ids": [remote_id],
        }

    monkeypatch.setattr(watcher, "_run_updater", run_updater)
    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert len(calls) == 1
    assert written[-1]["candidate_chtc_execution_ids"] == [remote_id]
    assert written[-1]["reported_chtc_execution_ids"] == [remote_id]


def test_watch_terminates_after_exact_nine_local_and_three_chtc_cells_are_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    campaign = _campaign(closed=local_ids, all_closed=True)
    written = _patch_watch_io(monkeypatch, campaign, reported=local_ids)
    chtc_ids = (
        "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
        "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
        "always_commutation_reduced",
        "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__"
        "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
        "no_lanes_always_commutation_reduced",
        watcher.snapshot.SW_ALWAYS_EXECUTION_ID,
    )
    monkeypatch.setattr(watcher, "_candidate_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(watcher, "_reported_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("all authenticated cells are already reported"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert written[-1]["status"] == "passed_all_twelve_hybrid_cells_refreshed"
    assert written[-1]["next_poll_seconds"] is None
    assert written[-1]["source_campaign"]["all_nine_local_cells_closed"] is True
    assert written[-1]["source_campaign"]["all_three_chtc_cells_refreshed"] is True
    assert written[-1]["source_campaign"]["all_twelve_hybrid_cells_refreshed"] is True


def test_same_local_id_refreshes_when_authenticated_k50_revision_replaces_k30(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    execution_id = "eligible-id"
    new_revision = "b" * 64
    campaign = _campaign(
        closed=(execution_id,),
        revisions={execution_id: new_revision},
    )
    written = _patch_watch_io(
        monkeypatch,
        campaign,
        reported=(execution_id,),
        reported_revisions={execution_id: "a" * 64},
    )
    calls: list[dict[str, object]] = []

    def run_updater(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "updated_existing_report_in_place",
            "page_count": 17,
            "preserved_page_count": 16,
            "pdf": {"sha256": SHA, "size_bytes": 10},
            "reported_local_execution_ids": [execution_id],
            "reported_local_evidence_revisions": {execution_id: new_revision},
            "reported_continuation_evidence_revision": "c" * 64,
            "reported_chtc_execution_ids": [],
        }

    monkeypatch.setattr(watcher, "_run_updater", run_updater)

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert len(calls) == 1
    assert written[-1]["reported_local_evidence_revisions"] == {
        execution_id: new_revision
    }


def test_nine_k30_plus_three_chtc_is_not_terminal_while_k50_is_pending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    revisions = {execution_id: "a" * 64 for execution_id in local_ids}
    campaign = _campaign(
        closed=local_ids,
        all_closed=True,
        revisions=revisions,
        all_continuations_closed=False,
        macro_terminal_authenticated=False,
        eligible_continuations=("eligible-id",),
    )
    written = _patch_watch_io(
        monkeypatch,
        campaign,
        reported=local_ids,
        reported_revisions=revisions,
    )
    chtc_ids = watcher.EXPECTED_CHTC_EXECUTION_IDS
    monkeypatch.setattr(watcher, "_candidate_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(watcher, "_reported_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("reported evidence has not changed"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert written[-1]["status"] == "watching_for_required_k50_continuations"
    assert written[-1]["next_poll_seconds"] == 30.0


def test_no_eligible_continuations_still_waits_for_macro_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    revisions = {execution_id: "a" * 64 for execution_id in local_ids}
    campaign = _campaign(
        closed=local_ids,
        all_closed=True,
        revisions=revisions,
        all_continuations_closed=True,
        macro_terminal_authenticated=False,
        eligible_continuations=(),
    )
    written = _patch_watch_io(
        monkeypatch,
        campaign,
        reported=local_ids,
        reported_revisions=revisions,
    )
    monkeypatch.setattr(
        watcher, "_candidate_chtc_execution_ids", lambda: watcher.EXPECTED_CHTC_EXECUTION_IDS
    )
    monkeypatch.setattr(
        watcher, "_reported_chtc_execution_ids", lambda: watcher.EXPECTED_CHTC_EXECUTION_IDS
    )
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("all required evidence is already reported"),
    )

    assert watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    ) == 0
    assert written[-1]["status"] == (
        "watching_for_authenticated_macro_terminal_receipt"
    )


def test_macro_terminal_receipt_revision_forces_provenance_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    revisions = {execution_id: "a" * 64 for execution_id in local_ids}
    new_continuation_revision = "d" * 64
    campaign = _campaign(
        closed=local_ids,
        all_closed=True,
        revisions=revisions,
        eligible_continuations=(local_ids[0],),
        continuation_revision=new_continuation_revision,
    )
    written = _patch_watch_io(
        monkeypatch,
        campaign,
        reported=local_ids,
        reported_revisions=revisions,
        reported_continuation_revision="c" * 64,
    )
    monkeypatch.setattr(
        watcher, "_candidate_chtc_execution_ids", lambda: watcher.EXPECTED_CHTC_EXECUTION_IDS
    )
    monkeypatch.setattr(
        watcher, "_reported_chtc_execution_ids", lambda: watcher.EXPECTED_CHTC_EXECUTION_IDS
    )
    calls: list[dict[str, object]] = []

    def run_updater(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "status": "updated_existing_report_in_place",
            "page_count": 17,
            "preserved_page_count": 16,
            "pdf": {"sha256": SHA, "size_bytes": 10},
            "reported_local_execution_ids": list(local_ids),
            "reported_local_evidence_revisions": revisions,
            "reported_continuation_evidence_revision": (
                new_continuation_revision
            ),
            "reported_chtc_execution_ids": list(
                watcher.EXPECTED_CHTC_EXECUTION_IDS
            ),
        }

    monkeypatch.setattr(watcher, "_run_updater", run_updater)

    assert watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    ) == 0
    assert len(calls) == 1
    assert written[-1]["status"] == "passed_all_twelve_hybrid_cells_refreshed"
    assert written[-1]["reported_continuation_evidence_revision"] == (
        new_continuation_revision
    )


def test_non_once_watch_exits_at_hybrid_terminal_without_sleeping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    campaign = _campaign(closed=local_ids, all_closed=True)
    written = _patch_watch_io(monkeypatch, campaign, reported=local_ids)
    chtc_ids = watcher.EXPECTED_CHTC_EXECUTION_IDS
    monkeypatch.setattr(watcher, "_candidate_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(watcher, "_reported_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("all authenticated cells are already reported"),
    )
    monkeypatch.setattr(
        watcher.time,
        "sleep",
        lambda _seconds: pytest.fail("terminal watcher must exit without sleeping"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=False,
    )

    assert result == 0
    assert written[-1]["status"] == "passed_all_twelve_hybrid_cells_refreshed"
    assert written[-1]["next_poll_seconds"] is None


def test_hybrid_terminal_compares_exact_id_sets_not_provenance_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    local_ids = tuple(f"local-{index}" for index in range(9))
    campaign = _campaign(closed=local_ids, all_closed=True)
    written = _patch_watch_io(
        monkeypatch,
        campaign,
        reported=tuple(reversed(local_ids)),
    )
    chtc_ids = watcher.EXPECTED_CHTC_EXECUTION_IDS
    monkeypatch.setattr(watcher, "_candidate_chtc_execution_ids", lambda: chtc_ids)
    monkeypatch.setattr(
        watcher,
        "_reported_chtc_execution_ids",
        lambda: tuple(reversed(chtc_ids)),
    )
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("exact sets are already fully reported"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=True,
    )

    assert result == 0
    assert written[-1]["status"] == "passed_all_twelve_hybrid_cells_refreshed"
    assert written[-1]["source_campaign"]["all_three_chtc_cells_refreshed"] is True
    assert written[-1]["source_campaign"]["all_twelve_hybrid_cells_refreshed"] is True


def test_terminal_failed_campaign_is_persisted_and_exits_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    campaign = _campaign(terminal_failed=True)
    written = _patch_watch_io(monkeypatch, campaign)
    monkeypatch.setattr(
        watcher,
        "_run_updater",
        lambda **_kwargs: pytest.fail("no completion growth exists"),
    )

    result = watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=False,
    )

    assert result == 2
    assert written[-1]["status"] == (
        "source_campaign_terminal_failed_no_active_supervisor_path"
    )
    assert written[-1]["next_poll_seconds"] is None


def test_reporting_authentication_failure_is_persisted_and_exits_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[dict[str, object]] = []
    monkeypatch.setattr(watcher, "_load_previous_status", lambda: None)
    monkeypatch.setattr(
        watcher,
        "_campaign_snapshot",
        lambda **_kwargs: (_ for _ in ()).throw(
            watcher.snapshot.UpdateError("continuation runtime drift")
        ),
    )

    def write_status(value: dict[str, object]) -> dict[str, object]:
        written.append(value)
        return value

    monkeypatch.setattr(watcher, "_write_status", write_status)

    assert watcher.watch(
        runtime_dir=tmp_path / "runtime",
        activation_dir=tmp_path / "activation",
        expected_adapter_sha256=SHA,
        poll_seconds=30.0,
        max_poll_seconds=300.0,
        once=False,
    ) == 1
    assert written[-1]["status"] == "watcher_authentication_failed"
    assert written[-1]["last_error"] == "continuation runtime drift"


def test_adaptive_polling_never_drops_below_thirty_seconds() -> None:
    fingerprint = "state"
    assert watcher._next_poll_seconds(
        previous=None,
        fingerprint=fingerprint,
        base=30.0,
        maximum=300.0,
    ) == 30.0
    assert watcher._next_poll_seconds(
        previous={
            "source_state_fingerprint": fingerprint,
            "next_poll_seconds": 30.0,
        },
        fingerprint=fingerprint,
        base=30.0,
        maximum=300.0,
    ) == 45.0
    assert watcher._next_poll_seconds(
        previous={
            "source_state_fingerprint": fingerprint,
            "next_poll_seconds": 290.0,
        },
        fingerprint=fingerprint,
        base=30.0,
        maximum=300.0,
    ) == 300.0
