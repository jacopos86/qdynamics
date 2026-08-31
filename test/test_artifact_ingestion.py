"""The incremental-ingestion transaction engine.

The corpus tests run against the sixty real Paper-II increment receipts, which
are the fixture set the engine must reproduce: reconciliation over them has to
report full coverage for every already-ingested cluster, or the engine would
re-ingest evidence the hand-executed transactions already recorded.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from pipelines.artifacts.ingestion import (
    PHASES,
    PhaseFailed,
    PhaseJournal,
    TransactionLock,
    TransactionLocked,
    TransactionSpec,
    load_receipts,
    newly_completed,
    receipt_covered_processes,
    run_ingestion_transaction,
)

ATTEMPTS_ROOT = Path("output/paper_ii_staged/attempts")


# --- reconciliation (pure) -------------------------------------------------


def test_newly_completed_is_the_contract_set_difference() -> None:
    receipts = [{"cluster_id": 9, "process_ids": [0, 1, 4]}]
    assert newly_completed(9, [0, 1, 2, 3, 4], receipts) == (2, 3)


def test_receipts_for_other_clusters_cover_nothing_here() -> None:
    receipts = [{"cluster_id": 8, "process_ids": [0, 1, 2]}]
    assert newly_completed(9, [0, 1], receipts) == (0, 1)


def test_no_receipts_means_everything_is_new() -> None:
    assert newly_completed(9, [3, 1, 2], []) == (1, 2, 3)


def test_receipt_without_process_ids_covers_nothing() -> None:
    """Unknowable coverage must not be treated as full coverage."""

    receipts = [{"cluster_id": 9}, {"cluster_id": 9, "process_ids": [1]}]
    assert newly_completed(9, [0, 1], receipts) == (0,)


def test_coverage_accumulates_across_receipts() -> None:
    receipts = [
        {"cluster_id": 9, "process_ids": [0]},
        {"cluster_id": 9, "process_ids": [1, 2]},
    ]
    covered = receipt_covered_processes(receipts)
    assert covered[9] == frozenset({0, 1, 2})


# --- the real corpus -------------------------------------------------------


@pytest.mark.skipif(not ATTEMPTS_ROOT.exists(), reason="no local attempt corpus")
def test_corpus_receipts_all_load_and_carry_process_ids() -> None:
    receipts = load_receipts(ATTEMPTS_ROOT)
    assert len(receipts) >= 50
    assert all(isinstance(r.get("process_ids"), list) for r in receipts)


@pytest.mark.skipif(not ATTEMPTS_ROOT.exists(), reason="no local attempt corpus")
def test_corpus_reconciliation_reports_full_coverage() -> None:
    """Replaying each cluster's own receipts must leave nothing newly completed."""

    receipts = load_receipts(ATTEMPTS_ROOT)
    covered = receipt_covered_processes(receipts)
    assert covered, "corpus produced no coverage at all"
    for cluster, processes in covered.items():
        assert newly_completed(cluster, processes, receipts) == ()


def test_unreadable_receipt_is_an_error_not_a_skip(tmp_path: Path) -> None:
    bad = tmp_path / "cluster_1/increments/x/evidence/increment_receipt.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="unreadable increment receipt"):
        load_receipts(tmp_path)


# --- lock ------------------------------------------------------------------


def test_lock_is_exclusive_and_reacquirable(tmp_path: Path) -> None:
    path = tmp_path / "transaction.lock"
    first = TransactionLock(lock_path=path, owner="agent-a").acquire()
    with pytest.raises(TransactionLocked, match="already owned"):
        TransactionLock(lock_path=path, owner="agent-b").acquire()
    first.release()
    TransactionLock(lock_path=path, owner="agent-b").acquire().release()


def test_lock_context_manager_releases_on_exit(tmp_path: Path) -> None:
    path = tmp_path / "transaction.lock"
    with TransactionLock(lock_path=path, owner="a"):
        assert path.exists()
    assert path.exists()
    TransactionLock(lock_path=path, owner="b").acquire().release()


def _acquire_lock_and_crash(path: str) -> None:
    TransactionLock(lock_path=Path(path), owner="crashing-child").acquire()
    raise SystemExit(0)


def test_lock_is_released_when_owner_process_exits(tmp_path: Path) -> None:
    path = tmp_path / "transaction.lock"
    process = multiprocessing.Process(
        target=_acquire_lock_and_crash, args=(str(path),)
    )
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0
    TransactionLock(lock_path=path, owner="recovery").acquire().release()


# --- journal ---------------------------------------------------------------


def test_journal_replays_done_phases(tmp_path: Path) -> None:
    journal = PhaseJournal(tmp_path / "j.jsonl")
    journal.record("reconcile", "done", {"n": 2})
    journal.record("transfer", "failed", {"guard": "sha_mismatch"})
    journal.record("transfer", "done", {"archive": "x.tgz"})
    done = journal.completed_phases()
    assert set(done) == {"reconcile", "transfer"}
    assert done["transfer"]["archive"] == "x.tgz"


def test_journal_rejects_double_done(tmp_path: Path) -> None:
    journal = PhaseJournal(tmp_path / "j.jsonl")
    journal.record("extract", "done", {})
    journal.record("extract", "done", {})
    with pytest.raises(ValueError, match="done\\s+twice|done twice"):
        journal.completed_phases()


def test_journal_rejects_unknown_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        PhaseJournal(tmp_path / "j.jsonl").record("teleport", "done", {})


# --- transaction engine ----------------------------------------------------


def _spec(tmp_path: Path, processes=(0, 1)) -> TransactionSpec:
    return TransactionSpec(
        campaign_id="unit_campaign",
        cluster_id=77,
        process_ids=tuple(processes),
        increment_root=tmp_path / "attempts/cluster_77/increments/procs0-1",
        owner="test-agent",
    )


def test_happy_path_runs_hooks_in_order_and_writes_receipt(tmp_path: Path) -> None:
    order: list[str] = []

    def hook(name):
        def run(context):
            order.append(name)
            return {"ran": name}
        return run

    spec = _spec(tmp_path)
    result = run_ingestion_transaction(
        spec, {p: hook(p) for p in ("reconcile", "transfer", "ledger_append")}
    )
    assert result.status == "ingested"
    assert order == ["reconcile", "transfer", "ledger_append"]  # canonical order
    receipt = json.loads(result.receipt_path.read_text())
    assert receipt["cluster_id"] == 77
    assert receipt["process_ids"] == [0, 1]
    assert receipt["ledger_append"] == {"ran": "ledger_append"}
    # engine-side coverage contract: this receipt now covers the delta
    assert newly_completed(77, [0, 1], [receipt]) == ()


def test_storage_guards_bracket_transfer_and_precede_extraction(tmp_path: Path) -> None:
    order: list[str] = []

    def hook(name):
        def run(_context):
            order.append(name)
            return {}
        return run

    phases = (
        "storage_guard_pre_transfer",
        "transfer",
        "storage_guard_pre_extract",
        "extract",
    )
    result = run_ingestion_transaction(
        _spec(tmp_path), {phase: hook(phase) for phase in phases}
    )
    assert result.status == "ingested"
    assert order == list(phases)


def test_failed_hook_writes_blocked_receipt_naming_the_guard(tmp_path: Path) -> None:
    def failing(_context):
        raise PhaseFailed("transfer", "sha256_mismatch", "expected abc, got def")

    result = run_ingestion_transaction(_spec(tmp_path), {"transfer": failing})
    assert result.status == "blocked"
    assert result.blocked_phase == "transfer"
    blocked = json.loads(result.receipt_path.read_text())
    assert blocked["failed_guard"] == "sha256_mismatch"
    assert blocked["process_ids"] == [0, 1]
    assert blocked["schema"] == "incremental_refresh_blocked_v1"


def test_blocked_delta_stays_newly_completed(tmp_path: Path) -> None:
    """A blocked receipt must not count as coverage: the delta is offered again.

    load_receipts globs only increment_receipt.json, so the blocked receipt in
    the same evidence dir is invisible to reconciliation -- verified here from
    the filesystem, not assumed from the glob.
    """

    def failing(_context):
        raise PhaseFailed("transfer", "sha256_mismatch")

    spec = _spec(tmp_path)
    result = run_ingestion_transaction(spec, {"transfer": failing})
    assert result.status == "blocked"
    receipts = load_receipts(tmp_path / "attempts")
    assert receipts == []                      # blocked receipt not picked up
    assert newly_completed(77, [0, 1], receipts) == (0, 1)


def test_resume_skips_completed_phases_and_replays_their_payloads(tmp_path: Path) -> None:
    calls: list[str] = []
    spec = _spec(tmp_path)

    def ok(name):
        def run(context):
            calls.append(name)
            return {"ran": name}
        return run

    def fail_once(context):
        calls.append("validate")
        raise PhaseFailed("validate", "achieved_config_mismatch")

    hooks = {"reconcile": ok("reconcile"), "validate": fail_once,
             "report_refresh": ok("report_refresh")}
    first = run_ingestion_transaction(spec, hooks)
    assert first.status == "blocked"
    assert calls == ["reconcile", "validate"]

    seen_context: dict = {}

    def validate_fixed(context):
        calls.append("validate2")
        seen_context.update(context)
        return {"validated": True}

    hooks["validate"] = validate_fixed
    second = run_ingestion_transaction(spec, hooks)
    assert second.status == "ingested"
    # reconcile did NOT re-run; its journaled payload was replayed into context
    assert calls == ["reconcile", "validate", "validate2", "report_refresh"]
    assert seen_context["reconcile"] == {"ran": "reconcile"}


def test_completed_transaction_is_an_idempotent_noop(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    count = {"n": 0}

    def hook(context):
        count["n"] += 1
        return {}

    first = run_ingestion_transaction(spec, {"ledger_append": hook})
    second = run_ingestion_transaction(spec, {"ledger_append": hook})
    assert first.status == second.status == "ingested"
    assert count["n"] == 1
    assert second.receipt_path == first.receipt_path


def test_empty_process_set_is_a_noop(tmp_path: Path) -> None:
    result = run_ingestion_transaction(_spec(tmp_path, processes=()), {})
    assert result.status == "noop"
    assert result.receipt_path is None


def test_unknown_hook_phase_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown hook phase"):
        run_ingestion_transaction(_spec(tmp_path), {"telport": lambda c: {}})


def test_concurrent_transaction_on_same_delta_is_locked_out(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    lock = TransactionLock(
        lock_path=spec.campaign_lock_path, owner="other-agent"
    ).acquire()
    try:
        with pytest.raises(TransactionLocked):
            run_ingestion_transaction(spec, {"reconcile": lambda c: {}})
    finally:
        lock.release()


def test_campaign_lock_blocks_overlapping_different_increment_roots(
    tmp_path: Path,
) -> None:
    first = _spec(tmp_path, processes=(0, 1))
    second = TransactionSpec(
        campaign_id=first.campaign_id,
        cluster_id=77,
        process_ids=(1, 2),
        increment_root=(
            tmp_path / "attempts/cluster_77/increments/procs1-2"
        ),
        owner="second-agent",
    )
    assert first.campaign_lock_path == second.campaign_lock_path
    lock = TransactionLock(
        lock_path=first.campaign_lock_path, owner="first-agent"
    ).acquire()
    try:
        with pytest.raises(TransactionLocked):
            run_ingestion_transaction(second, {"reconcile": lambda _context: {}})
    finally:
        lock.release()
