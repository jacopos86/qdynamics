r"""Paper-II hook bindings for the incremental-ingestion transaction engine.

`pipelines/artifacts/ingestion.py` owns ordering, locking, journaling, and
receipt assembly; this module supplies the Paper-II side effects as hooks.
Nothing here re-implements a step that already has a tested unit: archive
verification wraps the same checks the operating contract lists, ledger append
delegates to ``record_prepared_action_outputs`` (which validates achieved
configuration and computes exact-reference metrics internally -- that tested
boundary is deliberately not torn apart here), and the report refresh is an
injected callable so tests never compile the live PDF.

The one behavioral change from hand execution is the defensive re-check in the
``reconcile`` hook: coverage is recomputed *inside the transaction lock*, so
two supervisors racing on one scheduler snapshot cannot both ingest the same
delta -- the loser sees full coverage and gets a clean no-op instead of a
duplicate ledger append.
"""

from __future__ import annotations

import gzip
import hashlib
import zlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipelines.artifacts.ingestion import (
    Hook,
    PhaseFailed,
    TransactionSpec,
    load_receipts,
    newly_completed,
)
from pipelines.artifacts.storage_guard import assess_fetch_storage

PAPER_II_INGESTION_ADAPTER_V1 = "paper_ii_ingestion_adapter_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempts_root(spec: TransactionSpec) -> Path:
    # increment_root = .../attempts/cluster_<id>/increments/<name>
    # parents: [0] increments, [1] cluster_<id>, [2] attempts
    return spec.increment_root.parents[2]


def reconcile_hook(spec: TransactionSpec) -> Hook:
    """Re-check coverage inside the lock; refuse an already-covered delta."""

    def run(_context: Mapping[str, Any]) -> Mapping[str, Any]:
        receipts = load_receipts(_attempts_root(spec))
        fresh = newly_completed(spec.cluster_id, spec.process_ids, receipts)
        if tuple(fresh) != tuple(sorted(spec.process_ids)):
            covered = sorted(set(spec.process_ids) - set(fresh))
            raise PhaseFailed(
                "reconcile",
                "delta_already_covered",
                f"processes {covered} are covered by durable receipts; a "
                "concurrent transaction won the race",
            )
        return {
            "durable_receipt_count": len(receipts),
            "newly_completed": list(fresh),
        }

    return run


def storage_guard_hook(
    archive_path: Path,
    *,
    phase: str,
    free_floor_bytes: int | None = None,
) -> Hook:
    if phase not in {"storage_guard_pre_transfer", "storage_guard_pre_extract"}:
        raise ValueError(f"unsupported storage-guard phase {phase!r}")

    def run(context: Mapping[str, Any]) -> Mapping[str, Any]:
        archive_bytes: int | None = None
        if archive_path.is_file():
            archive_bytes = int(archive_path.stat().st_size)
        remote = context.get("remote_package")
        if archive_bytes is None and isinstance(remote, Mapping):
            value = remote.get("archive_bytes")
            if value is not None:
                archive_bytes = int(value)
        if archive_bytes is None:
            raise PhaseFailed(phase, "archive_size_unknown", str(archive_path))
        probe = archive_path.parent
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        kwargs: dict[str, Any] = {
            "free_bytes": int(usage.free),
            "archive_bytes": archive_bytes,
        }
        if free_floor_bytes is not None:
            kwargs["free_floor_bytes"] = int(free_floor_bytes)
        assessment = assess_fetch_storage(**kwargs)
        if not bool(assessment.allowed):
            raise PhaseFailed(
                phase,
                "free_space_floor",
                json.dumps(getattr(assessment, "__dict__", {}), default=str),
            )
        return {
            "free_bytes": int(usage.free),
            "archive_bytes": kwargs["archive_bytes"],
            "guard_phase": phase,
        }

    return run


def archive_verification_hook(
    archive_path: Path, *, expected_sha256: str | None = None
) -> Hook:
    """Size, SHA-256, gzip integrity, and tar readability of the fetched delta."""

    def run(_context: Mapping[str, Any]) -> Mapping[str, Any]:
        if not archive_path.is_file():
            raise PhaseFailed("transfer", "archive_missing", str(archive_path))
        digest = _sha256(archive_path)
        if expected_sha256 is not None and digest != str(expected_sha256):
            raise PhaseFailed(
                "transfer",
                "sha256_mismatch",
                f"expected {expected_sha256}, got {digest}",
            )
        try:
            with gzip.open(archive_path, "rb") as handle:
                while handle.read(1 << 20):
                    pass
        except (OSError, EOFError, zlib.error) as exc:
            # header damage raises BadGzipFile (an OSError); mid-stream
            # corruption raises zlib.error, which is not an OSError.
            raise PhaseFailed("transfer", "gzip_integrity", str(exc))
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                member_count = sum(1 for _ in archive)
        except tarfile.TarError as exc:
            raise PhaseFailed("transfer", "tar_integrity", str(exc))
        return {
            "archive": str(archive_path),
            "archive_bytes": int(archive_path.stat().st_size),
            "archive_sha256": digest,
            "archive_member_count": int(member_count),
        }

    return run


def extraction_hook(archive_path: Path, extraction_root: Path) -> Hook:
    """Extract regular files/directories only, with path confinement."""

    def run(_context: Mapping[str, Any]) -> Mapping[str, Any]:
        extraction_root.mkdir(parents=True, exist_ok=True)
        root = extraction_root.resolve()
        extracted = 0
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                target = (root / member.name).resolve()
                if not target.is_relative_to(root):
                    raise PhaseFailed(
                        "extract", "path_escape", member.name
                    )
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise PhaseFailed("extract", "unsafe_tar_member", member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    source = archive.extractfile(member)
                    if source is None:
                        raise PhaseFailed(
                            "extract", "tar_member_unreadable", member.name
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                else:
                    raise PhaseFailed("extract", "unsafe_tar_member", member.name)
                extracted += 1
        return {"extraction_root": str(root), "extracted_members": extracted}

    return run


def ledger_append_hook(
    *,
    campaign: Any,
    ledger_path: Path,
    action_rows: Sequence[Mapping[str, Any]],
    seed_path: Path,
    numerics: str,
    gate: str,
    search_profile: str,
    process_ids: Sequence[int] | None = None,
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
    record_terminal_logical_resources: bool = True,
) -> Hook:
    """Validate, exact-evaluate, and append -- via the existing tested unit.

    ``record_prepared_action_outputs`` performs achieved-configuration
    validation and exact-reference evaluation internally before appending.
    Splitting those into separate engine phases would mean re-plumbing a
    tested boundary during a migration; if that split is ever wanted, it is
    its own change with its own tests.
    """

    def run(context: Mapping[str, Any]) -> Mapping[str, Any]:
        from pipelines.time_dynamics.diagnostics.terminal_logical_graph_resources import (
            TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME,
            build_terminal_logical_graph_resource_record,
            write_terminal_logical_graph_resource_record,
        )
        from pipelines.time_dynamics.staged_campaign import CampaignLedger
        from pipelines.time_dynamics.staged_campaign_artifacts import (
            record_prepared_action_outputs,
        )

        extract_payload = context.get("extract") or {}
        artifact_root = extract_payload.get("extraction_root")
        if not artifact_root:
            raise PhaseFailed(
                "ledger_append", "no_extraction_root",
                "extract phase did not record an extraction root",
            )
        declared_processes = (
            () if process_ids is None else tuple(int(value) for value in process_ids)
        )
        if (
            len(declared_processes) != len(action_rows)
            or len(set(declared_processes)) != len(declared_processes)
        ):
            raise PhaseFailed(
                "ledger_append",
                "process_coverage_mismatch",
                (
                    f"processes={len(declared_processes)}, "
                    f"unique_processes={len(set(declared_processes))}, "
                    f"actions={len(action_rows)}"
                ),
            )
        ledger = (
            CampaignLedger.read(ledger_path)
            if ledger_path.is_file()
            else CampaignLedger()
        )
        before_count = len(ledger.records)
        before_sha = _sha256(ledger_path) if ledger_path.is_file() else None
        ledger, receipts = record_prepared_action_outputs(
            campaign=campaign,
            ledger=ledger,
            action_rows=tuple(action_rows),
            artifact_root=Path(artifact_root),
            seed_path=seed_path,
            numerics=str(numerics),
            gate=str(gate),
            search_profile=str(search_profile),
            reference_substeps=int(reference_substeps),
            reference_store_root=reference_store_root,
        )
        if len(receipts) != len(action_rows):
            raise PhaseFailed(
                "ledger_append",
                "process_coverage_mismatch",
                (
                    f"actions={len(action_rows)}, receipts={len(receipts)}"
                ),
            )
        dispositions: list[dict[str, Any]] = []
        for process_id, action_row, receipt in zip(
            declared_processes, action_rows, receipts, strict=True
        ):
            action_id = str(action_row.get("action_id") or "")
            if str(receipt.get("action_id") or "") != action_id:
                raise PhaseFailed(
                    "ledger_append",
                    "action_coverage_mismatch",
                    (
                        f"process {process_id}: action row {action_id!r}, "
                        f"receipt {receipt.get('action_id')!r}"
                    ),
                )
            run_path = Path(str(receipt.get("run_path") or ""))
            run_sha256 = str(receipt.get("run_sha256") or "")
            if (
                not run_path.is_file()
                or len(run_sha256) != 64
                or _sha256(run_path) != run_sha256
            ):
                raise PhaseFailed(
                    "ledger_append",
                    "run_coverage_unverifiable",
                    f"process {process_id}: {run_path}",
                )
            resource_artifact: dict[str, str] | None = None
            if record_terminal_logical_resources:
                resource_path = run_path.with_name(
                    TERMINAL_LOGICAL_GRAPH_RESOURCE_FILENAME
                )
                try:
                    resource_payload = build_terminal_logical_graph_resource_record(
                        run_path
                    )
                    write_terminal_logical_graph_resource_record(
                        resource_path, resource_payload
                    )
                except Exception as exc:
                    raise PhaseFailed(
                        "ledger_append",
                        "terminal_logical_resource_invalid",
                        f"process {process_id}: {exc}",
                    ) from exc
                resource_artifact = {
                    "path": str(resource_path),
                    "sha256": _sha256(resource_path),
                    "resource_sha256": str(resource_payload["resource_sha256"]),
                }
            disposition = {
                "process_id": int(process_id),
                "action_id": action_id,
                "run_path": str(run_path),
                "run_sha256": run_sha256,
                "status": str(receipt.get("status") or ""),
            }
            if resource_artifact is not None:
                disposition["terminal_logical_graph_resources"] = resource_artifact
            if receipt.get("ledger_status") not in (None, ""):
                disposition["ledger_status"] = str(receipt["ledger_status"])
            dispositions.append(disposition)
        # Commit scientific state only after every process/action/run binding
        # passes. A malformed adapter result therefore leaves the prior ledger
        # byte-for-byte intact and this phase remains safely retryable.
        ledger.write(ledger_path)
        ledger_after_sha256 = _sha256(ledger_path)
        return {
            "ledger": str(ledger_path),
            "ledger_record_count_before": before_count,
            "ledger_record_count_after": len(ledger.records),
            "ledger_before_sha256": before_sha,
            "ledger_after_sha256": ledger_after_sha256,
            "action_receipts": list(receipts),
            "action_dispositions": dispositions,
            "scientific_state_after": {
                "path": str(ledger_path),
                "sha256": ledger_after_sha256,
            },
        }

    return run


def report_refresh_hook(refresh: Callable[[], Mapping[str, Any]]) -> Hook:
    """Wrap a report-refresh callable; its payload lands in the receipt.

    Injected rather than hardcoded so tests never compile the live PDF, and so
    a lane can bind promotion_preview, the staged report, or both.
    """

    def run(_context: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            payload = dict(refresh())
        except PhaseFailed:
            raise
        except Exception as exc:  # a report crash is a named guard, not a trace
            raise PhaseFailed("report_refresh", "report_builder_error", str(exc))
        return payload

    return run


def build_paper_ii_hooks(
    spec: TransactionSpec,
    *,
    campaign: Any,
    ledger_path: Path,
    archive_path: Path,
    action_rows: Sequence[Mapping[str, Any]],
    seed_path: Path,
    numerics: str,
    gate: str,
    search_profile: str,
    expected_archive_sha256: str | None = None,
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
    report_refresh: Callable[[], Mapping[str, Any]] | None = None,
    free_floor_bytes: int | None = None,
    record_terminal_logical_resources: bool = True,
) -> dict[str, Hook]:
    """The full Paper-II hook set, in the engine's canonical phase order."""

    hooks: dict[str, Hook] = {
        "reconcile": reconcile_hook(spec),
        "storage_guard_pre_transfer": storage_guard_hook(
            archive_path,
            phase="storage_guard_pre_transfer",
            free_floor_bytes=free_floor_bytes,
        ),
        "transfer": archive_verification_hook(
            archive_path, expected_sha256=expected_archive_sha256
        ),
        "storage_guard_pre_extract": storage_guard_hook(
            archive_path,
            phase="storage_guard_pre_extract",
            free_floor_bytes=free_floor_bytes,
        ),
        "extract": extraction_hook(
            archive_path, spec.evidence_dir / "extracted"
        ),
        "ledger_append": ledger_append_hook(
            campaign=campaign,
            ledger_path=ledger_path,
            action_rows=action_rows,
            seed_path=seed_path,
            numerics=numerics,
            gate=gate,
            search_profile=search_profile,
            process_ids=spec.process_ids,
            reference_substeps=reference_substeps,
            reference_store_root=reference_store_root,
            record_terminal_logical_resources=record_terminal_logical_resources,
        ),
    }
    if report_refresh is not None:
        hooks["report_refresh"] = report_refresh_hook(report_refresh)
    return hooks


__all__ = [
    "PAPER_II_INGESTION_ADAPTER_V1",
    "archive_verification_hook",
    "build_paper_ii_hooks",
    "extraction_hook",
    "ledger_append_hook",
    "reconcile_hook",
    "report_refresh_hook",
    "storage_guard_hook",
]
