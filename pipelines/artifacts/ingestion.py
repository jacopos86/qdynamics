r"""Incremental-ingestion transaction engine for scheduler evidence streams.

The operating contract (`agent_guidance/time-dynamics/paper-ii-overnight-campaign.md`)
defines a twelve-step snapshot-to-report transaction that, until this module,
existed only as prose and was executed by hand -- sixty times, leaving sixty
increment receipts in ten distinct key-set variants.  Schema drift of that kind
is what hand execution produces, and the steps most likely to be skipped under
time pressure (integrity checks, ordering, receipt assembly) are exactly the
ones with no code behind them.

This engine owns the parts that must never vary:

* **Reconciliation** -- ``newly_completed = completed - receipt_covered`` as a
  pure function over durable receipts, keyed by ``(cluster_id, process_id)``.
* **A campaign-wide transaction lock** spanning reconcile through ledger
  append, so overlapping deltas and different clusters cannot race on one
  shared ledger.
* **A phase journal** -- append-only JSONL, one line per completed phase, so an
  interrupted transaction resumes at the failed phase instead of restarting or
  double-executing.
* **Receipt assembly** -- one machine-generated final receipt per transaction,
  or a precise ``incremental_refresh_blocked`` receipt naming the failed phase
  and guard.  "Completed but awaiting ingestion" is not a state this engine can
  emit.

The side-effecting steps -- remote delta packaging, transfer, extraction,
artifact validation, ledger append, report refresh, manuscript assets -- are
**injected hooks**.  Paper II binds its existing functions; another paper lane
binds its own.  The engine never reaches around a hook, and a hook's failure is
a blocked receipt, not a stack trace and a half-finished transaction.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

INGESTION_ENGINE_V1 = "incremental_ingestion_engine_v1"
INGESTION_RECEIPT_SCHEMA_V3 = "incremental_ingestion_receipt_v3"
BLOCKED_RECEIPT_SCHEMA_V1 = "incremental_refresh_blocked_v1"

# The canonical phase order of the snapshot-to-report transaction.  A hook set
# need not populate every phase (a local campaign has no remote transfer), but
# executed phases always run in this order and the journal records no others.
PHASES: tuple[str, ...] = (
    "reconcile",
    "remote_package",
    "storage_guard_pre_transfer",
    "transfer",
    "storage_guard_pre_extract",
    "extract",
    "validate",
    "evaluate",
    "ledger_append",
    "report_refresh",
    "manuscript_assets",
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Reconciliation (pure)
# ---------------------------------------------------------------------------


def receipt_covered_processes(
    receipts: Iterable[Mapping[str, Any]],
) -> dict[int, frozenset[int]]:
    """``cluster_id -> covered process ids`` from durable receipts.

    A receipt without ``process_ids`` covers nothing rather than everything:
    treating an unreadable receipt as full coverage would silently drop
    completed evidence, which is the failure the whole contract exists to
    prevent.
    """

    covered: dict[int, set[int]] = {}
    for receipt in receipts:
        cluster = receipt.get("cluster_id")
        processes = receipt.get("process_ids")
        if cluster is None or not isinstance(processes, (list, tuple)):
            continue
        covered.setdefault(int(cluster), set()).update(int(p) for p in processes)
    return {k: frozenset(v) for k, v in covered.items()}


def newly_completed(
    cluster_id: int,
    completed_process_ids: Iterable[int],
    receipts: Iterable[Mapping[str, Any]],
) -> tuple[int, ...]:
    """The contract's central rule, as a pure function.

    ``newly_completed = completed process IDs - process IDs in durable
    incremental receipts``, for one cluster.  Receipts for other clusters never
    cover this one's processes.
    """

    covered = receipt_covered_processes(receipts).get(int(cluster_id), frozenset())
    return tuple(
        sorted(set(int(p) for p in completed_process_ids) - set(covered))
    )


def load_receipts(attempts_root: str | Path) -> list[dict[str, Any]]:
    """Every durable increment receipt under an attempts root.

    Layout: ``<root>/cluster_<id>/increments/<name>/evidence/increment_receipt.json``.
    Unreadable files are surfaced, not skipped: a corrupt receipt makes coverage
    unknowable for its processes, and pretending otherwise re-ingests or drops
    evidence depending on the direction of the corruption.
    """

    root = Path(attempts_root)
    receipts: list[dict[str, Any]] = []
    for path in sorted(root.glob("cluster_*/increments/*/evidence/increment_receipt.json")):
        try:
            receipts.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unreadable increment receipt {path}: {exc}. Coverage is "
                "unknowable until it is repaired or moved aside explicitly."
            ) from exc
    return receipts


# ---------------------------------------------------------------------------
# Transaction lock
# ---------------------------------------------------------------------------


class TransactionLocked(RuntimeError):
    """Another transaction owns this delta."""


@dataclass
class TransactionLock:
    """Crash-releasing advisory lock for one campaign/cluster delta.

    The lockfile is durable owner metadata, while the kernel ``flock`` is the
    authority.  A killed process or reboot releases the lock automatically;
    the next supervisor may therefore resume the phase journal without a
    human deleting a stale ``O_EXCL`` sentinel.
    """

    lock_path: Path
    owner: str
    _held: bool = field(default=False, repr=False)
    _handle: Any = field(default=None, repr=False)

    def acquire(self) -> "TransactionLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(
                self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            self._handle.seek(0)
            existing = self._handle.read() or "(unreadable)"
            self._handle.close()
            self._handle = None
            raise TransactionLocked(
                f"delta already owned: {self.lock_path} holds {existing!r}"
            )
        self._handle.seek(0)
        self._handle.truncate()
        json.dump(
            {"owner": self.owner, "pid": os.getpid(), "acquired": _utc_now()},
            self._handle,
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._held = True
        return self

    def release(self) -> None:
        if self._held:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
            self._held = False

    def __enter__(self) -> "TransactionLock":
        return self.acquire()

    def __exit__(self, *_exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Phase journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JournalEntry:
    phase: str
    status: str  # "done" | "failed"
    recorded_at: str
    payload: Mapping[str, Any]


class PhaseJournal:
    """Append-only journal making an interrupted transaction resumable.

    One JSONL line per phase outcome.  Resume semantics: phases recorded
    ``done`` are skipped with their recorded payload replayed to later hooks;
    the first phase not recorded ``done`` runs next.  A ``failed`` line does
    not block a retry of the same phase -- it documents the failure.
    """

    def __init__(self, journal_path: str | Path):
        self.path = Path(journal_path)

    def entries(self) -> list[JournalEntry]:
        if not self.path.exists():
            return []
        out: list[JournalEntry] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(
                JournalEntry(
                    phase=str(row["phase"]),
                    status=str(row["status"]),
                    recorded_at=str(row.get("recorded_at", "")),
                    payload=dict(row.get("payload", {})),
                )
            )
        return out

    def completed_phases(self) -> dict[str, Mapping[str, Any]]:
        done: dict[str, Mapping[str, Any]] = {}
        for entry in self.entries():
            if entry.status == "done":
                if entry.phase in done:
                    raise ValueError(
                        f"journal {self.path} records phase {entry.phase!r} done "
                        "twice; a transaction must not re-execute a completed "
                        "phase, so this journal is corrupt."
                    )
                done[entry.phase] = entry.payload
        return done

    def record(self, phase: str, status: str, payload: Mapping[str, Any]) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; canonical order is {PHASES}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "phase": phase,
                        "status": status,
                        "recorded_at": _utc_now(),
                        "payload": dict(payload),
                    },
                    sort_keys=True,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# The transaction
# ---------------------------------------------------------------------------


class PhaseFailed(RuntimeError):
    def __init__(self, phase: str, guard: str, detail: str = ""):
        super().__init__(f"{phase}: {guard}" + (f" ({detail})" if detail else ""))
        self.phase = str(phase)
        self.guard = str(guard)
        self.detail = str(detail)


# A hook receives the accumulated context (immutable view) and returns a JSON
# payload merged into the context under its phase name.
Hook = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class TransactionSpec:
    campaign_id: str
    cluster_id: int
    process_ids: tuple[int, ...]
    increment_root: Path         # .../attempts/cluster_<id>/increments/<name>
    owner: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def evidence_dir(self) -> Path:
        return self.increment_root / "evidence"

    @property
    def campaign_lock_path(self) -> Path:
        """One lock for every increment sharing this campaign attempts root."""

        # increment_root = attempts/cluster_<id>/increments/<increment>
        return self.increment_root.parents[2] / "ingestion.lock"


@dataclass(frozen=True)
class TransactionResult:
    status: str                  # "ingested" | "blocked" | "noop"
    receipt_path: Path | None
    receipt: Mapping[str, Any] | None
    blocked_phase: str | None = None


def run_ingestion_transaction(
    spec: TransactionSpec,
    hooks: Mapping[str, Hook],
) -> TransactionResult:
    """Execute (or resume) one snapshot-to-report transaction.

    Ordering, locking, journaling, and receipt assembly are owned here; every
    side effect lives in ``hooks``.  Unknown hook names are rejected rather than
    ignored -- a misspelled hook silently skipping its phase is precisely the
    hand-execution failure this engine replaces.
    """

    unknown = set(hooks) - set(PHASES)
    if unknown:
        raise ValueError(f"unknown hook phase(s) {sorted(unknown)}; valid: {PHASES}")
    if not spec.process_ids:
        return TransactionResult(status="noop", receipt_path=None, receipt=None)

    receipt_path = spec.evidence_dir / "increment_receipt.json"
    if receipt_path.exists():
        # Idempotence: a completed transaction is a no-op, never a re-run.
        return TransactionResult(
            status="ingested",
            receipt_path=receipt_path,
            receipt=json.loads(receipt_path.read_text()),
        )

    journal = PhaseJournal(spec.evidence_dir / "transaction_journal.jsonl")
    lock = TransactionLock(
        lock_path=spec.campaign_lock_path, owner=spec.owner
    )
    with lock:
        context: dict[str, Any] = {
            "engine": INGESTION_ENGINE_V1,
            "campaign_id": spec.campaign_id,
            "cluster_id": int(spec.cluster_id),
            "process_ids": [int(p) for p in spec.process_ids],
            "increment_root": str(spec.increment_root),
            **dict(spec.extra),
        }
        completed = journal.completed_phases()
        for phase, payload in completed.items():
            context[phase] = dict(payload)

        for phase in PHASES:
            if phase in completed:
                continue
            hook = hooks.get(phase)
            if hook is None:
                continue
            try:
                payload = dict(hook(dict(context)))
            except PhaseFailed as failure:
                journal.record(phase, "failed", {"guard": failure.guard,
                                                 "detail": failure.detail})
                blocked = _write_blocked_receipt(spec, phase, failure)
                return TransactionResult(
                    status="blocked",
                    receipt_path=blocked,
                    receipt=json.loads(blocked.read_text()),
                    blocked_phase=phase,
                )
            journal.record(phase, "done", payload)
            context[phase] = payload

        receipt = _assemble_receipt(spec, context)
        spec.evidence_dir.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
        return TransactionResult(
            status="ingested", receipt_path=receipt_path, receipt=receipt
        )


def _write_blocked_receipt(
    spec: TransactionSpec, phase: str, failure: PhaseFailed
) -> Path:
    """The contract's only permitted early exit: a named, process-scoped guard."""

    spec.evidence_dir.mkdir(parents=True, exist_ok=True)
    path = spec.evidence_dir / "incremental_refresh_blocked.json"
    path.write_text(
        json.dumps(
            {
                "schema": BLOCKED_RECEIPT_SCHEMA_V1,
                "campaign_id": spec.campaign_id,
                "cluster_id": int(spec.cluster_id),
                "process_ids": [int(p) for p in spec.process_ids],
                "blocked_phase": phase,
                "failed_guard": failure.guard,
                "detail": failure.detail,
                "recorded_at": _utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path


def _assemble_receipt(
    spec: TransactionSpec, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Engine-owned receipt core; hook payloads carry the lane-specific fields.

    The engine guarantees the coverage-critical fields (schema, campaign,
    cluster, process ids, timestamps, per-phase payloads); everything a lane
    adds (archive hashes, ledger digests, report digests) arrives through its
    hooks' payloads and is preserved under the phase name that produced it.
    """

    receipt: dict[str, Any] = {
        "schema": INGESTION_RECEIPT_SCHEMA_V3,
        "engine": INGESTION_ENGINE_V1,
        "campaign_id": spec.campaign_id,
        "cluster_id": int(spec.cluster_id),
        "process_ids": [int(p) for p in spec.process_ids],
        "recorded_at": _utc_now(),
        "status": "INGESTED",
    }
    for phase in PHASES:
        if phase in context and isinstance(context[phase], Mapping):
            receipt[phase] = dict(context[phase])
    for key, value in context.items():
        if key not in receipt and key not in PHASES and key not in (
            "engine", "increment_root"
        ):
            receipt[key] = value
    return receipt


__all__ = [
    "BLOCKED_RECEIPT_SCHEMA_V1",
    "INGESTION_ENGINE_V1",
    "INGESTION_RECEIPT_SCHEMA_V3",
    "PHASES",
    "PhaseFailed",
    "PhaseJournal",
    "TransactionLock",
    "TransactionLocked",
    "TransactionResult",
    "TransactionSpec",
    "load_receipts",
    "newly_completed",
    "receipt_covered_processes",
    "run_ingestion_transaction",
]
