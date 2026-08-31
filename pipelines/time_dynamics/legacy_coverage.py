"""Strong, immutable normalization for pre-v3 Paper-II increment receipts.

Early streamed-ingestion receipts did not store the process/action/run digest
tuple that the acquisition supervisor now requires.  This module never edits
those receipts.  It reconstructs the tuple only when the original Condor
submission package, receipt, extracted run, and current scientific record all
prove the same identity, then writes a sidecar index binding every source by
SHA-256.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


LEGACY_COVERAGE_SCHEMA_V1 = "time_dynamics_legacy_coverage_index_v1"
_ACCEPTED_RECEIPT_SCHEMAS = frozenset(
    {
        "paper_ii_incremental_ingestion_receipt_v1",
        "paper_ii_incremental_ingestion_receipt_v2",
    }
)
_ACCEPTED_RECEIPT_STATUSES = frozenset(
    {
        "INGESTED_AND_REPORTED",
        "INGESTED_RECLASSIFIED_AND_REPORTED",
        "INGESTED_WITH_SAFEGUARDS_AND_REPORTED",
    }
)
_ACCEPTED_DISPOSITIONS = frozenset({"recorded", "recorded_safeguard"})


class LegacyCoverageError(ValueError):
    """A legacy coverage claim cannot be proven without an assumption."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyCoverageError(f"{label} is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyCoverageError(f"{label} is not a JSON object: {path}")
    return dict(value)


def _relative_file(root: Path, path: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise LegacyCoverageError(f"{label} escapes repository: {path}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise LegacyCoverageError(f"{label} is missing or empty: {resolved}")
    return relative.as_posix()


def _submission_packages(chtc_root: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(chtc_root.glob("**/submission_receipt.json")):
        try:
            value = _read_object(path, "submission receipt")
            cluster = int(value["cluster_id"])
        except (KeyError, TypeError, ValueError, LegacyCoverageError):
            continue
        if cluster in found and found[cluster].resolve() != path.resolve():
            raise LegacyCoverageError(
                f"cluster {cluster} has multiple local submission receipts"
            )
        found[cluster] = path
    return found


def _bound_action_map(
    *, root: Path, submission_path: Path, cluster_id: int
) -> tuple[list[str], dict[str, str]]:
    receipt = _read_object(submission_path, "submission receipt")
    if int(receipt.get("cluster_id", -1)) != cluster_id:
        raise LegacyCoverageError(f"submission cluster mismatch: {submission_path}")
    if receipt.get("status") != "SUBMITTED":
        raise LegacyCoverageError(f"submission was not accepted: {submission_path}")
    package = receipt.get("package")
    audit_binding: dict[str, str] = {}
    if not isinstance(package, Mapping):
        audit_raw = receipt.get("local_submission_audit")
        if not isinstance(audit_raw, Mapping):
            raise LegacyCoverageError(
                f"submission has no package binding: {submission_path}"
            )
        audit_path = root / str(audit_raw.get("path") or "")
        if _sha256_file(audit_path) != audit_raw.get("sha256"):
            raise LegacyCoverageError(
                f"local submission audit digest mismatch: {audit_path}"
            )
        audit = _read_object(audit_path, "local submission audit")
        package = {
            "manifest_sha256": audit.get("package_manifest_sha256"),
            "submit_descriptor_sha256": audit.get("submit_descriptor_sha256"),
        }
        audit_binding = {
            "local_submission_audit": _relative_file(
                root, audit_path, "local submission audit"
            ),
            "local_submission_audit_sha256": _sha256_file(audit_path),
        }
    package_root = submission_path.parent
    manifest_path = package_root / "input/manifest.json"
    action_ids_path = package_root / "input/action_ids.txt"
    descriptors = sorted(package_root.glob("*.sub"))
    if len(descriptors) != 1:
        raise LegacyCoverageError(
            f"submission must have exactly one descriptor: {package_root}"
        )
    descriptor_path = descriptors[0]
    expected_manifest_sha = str(package.get("manifest_sha256") or "")
    expected_descriptor_sha = str(package.get("submit_descriptor_sha256") or "")
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise LegacyCoverageError(f"submission manifest digest mismatch: {manifest_path}")
    if _sha256_file(descriptor_path) != expected_descriptor_sha:
        raise LegacyCoverageError(f"submit descriptor digest mismatch: {descriptor_path}")
    manifest = _read_object(manifest_path, "submission manifest")
    actions = manifest.get("actions")
    if not isinstance(actions, list):
        raise LegacyCoverageError(f"submission manifest has no actions: {manifest_path}")
    manifest_ids = [
        str(row.get("action_id") or "") if isinstance(row, Mapping) else ""
        for row in actions
    ]
    action_ids = [line.strip() for line in action_ids_path.read_text().splitlines()]
    if not action_ids or action_ids != manifest_ids or any(not item for item in action_ids):
        raise LegacyCoverageError(
            f"Condor action-id order differs from the bound manifest: {package_root}"
        )
    submitted_count = receipt.get("submitted_jobs", receipt.get("action_count", -1))
    if len(action_ids) != int(submitted_count):
        raise LegacyCoverageError(f"submitted job count mismatch: {submission_path}")
    descriptor_text = descriptor_path.read_text(encoding="utf-8")
    if "queue action_id from" not in descriptor_text or "input/action_ids.txt" not in descriptor_text:
        raise LegacyCoverageError(
            f"descriptor does not bind ProcId order to action_ids.txt: {descriptor_path}"
        )
    return action_ids, {
        "submission_receipt": _relative_file(root, submission_path, "submission receipt"),
        "submission_receipt_sha256": _sha256_file(submission_path),
        "package_manifest": _relative_file(root, manifest_path, "package manifest"),
        "package_manifest_sha256": expected_manifest_sha,
        "action_ids": _relative_file(root, action_ids_path, "action-id map"),
        "action_ids_sha256": _sha256_file(action_ids_path),
        "submit_descriptor": _relative_file(root, descriptor_path, "submit descriptor"),
        "submit_descriptor_sha256": expected_descriptor_sha,
        **audit_binding,
    }


def _run_map(receipt_path: Path) -> dict[str, Path]:
    roots = (receipt_path.parents[1] / "extracted", receipt_path.parent / "extracted")
    runs: dict[str, Path] = {}
    for extraction_root in roots:
        if not extraction_root.is_dir():
            continue
        for run in sorted(extraction_root.rglob("run.json")):
            action_id = run.parent.name
            previous = runs.get(action_id)
            if previous is not None and previous.resolve() != run.resolve():
                raise LegacyCoverageError(
                    f"legacy increment repeats run for {action_id}: {receipt_path}"
                )
            runs[action_id] = run
    return runs


def build_legacy_coverage_index(
    *,
    repository_root: str | Path,
    lane_id: str,
    attempts_root: str | Path,
    chtc_root: str | Path,
    scientific_state: str | Path,
    decision_handoff_sha256: str,
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Build a strong sidecar without modifying any historical receipt."""

    root = Path(repository_root).resolve()
    attempts = Path(attempts_root).resolve()
    packages = _submission_packages(Path(chtc_root).resolve())
    ledger_path = Path(scientific_state).resolve()
    ledger = _read_object(ledger_path, "scientific state")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise LegacyCoverageError("scientific state has no records list")
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if isinstance(record, Mapping) and record.get("run_path"):
            key = str(record["run_path"])
            if key in records_by_path:
                raise LegacyCoverageError(f"scientific state repeats run_path {key}")
            records_by_path[key] = record

    rows: list[dict[str, Any]] = []
    sources: dict[int, dict[str, str]] = {}
    receipt_paths = sorted(
        attempts.glob("cluster_*/increments/*/evidence/increment_receipt.json")
    )
    for receipt_path in receipt_paths:
        receipt = _read_object(receipt_path, "increment receipt")
        if receipt.get("schema") not in _ACCEPTED_RECEIPT_SCHEMAS:
            continue
        if receipt.get("status") not in _ACCEPTED_RECEIPT_STATUSES:
            raise LegacyCoverageError(
                f"legacy receipt has non-covering status: {receipt_path}"
            )
        cluster_id = int(receipt.get("cluster_id", -1))
        submission_path = packages.get(cluster_id)
        if submission_path is None:
            raise LegacyCoverageError(
                f"cluster {cluster_id} has no bound local submission package"
            )
        action_ids, source = _bound_action_map(
            root=root, submission_path=submission_path, cluster_id=cluster_id
        )
        prior_source = sources.setdefault(cluster_id, source)
        if prior_source != source:
            raise LegacyCoverageError(f"cluster {cluster_id} source binding changed")
        processes_raw = receipt.get("process_ids")
        if not isinstance(processes_raw, list) or not processes_raw:
            raise LegacyCoverageError(f"receipt has no process_ids: {receipt_path}")
        processes = [int(value) for value in processes_raw]
        if len(set(processes)) != len(processes):
            raise LegacyCoverageError(f"receipt repeats a process: {receipt_path}")
        run_map = _run_map(receipt_path)
        dispositions = receipt.get("action_dispositions")
        disposition_by_action: dict[str, Mapping[str, Any]] = {}
        if dispositions is not None:
            if not isinstance(dispositions, list):
                raise LegacyCoverageError(
                    f"receipt action_dispositions is not a list: {receipt_path}"
                )
            for disposition in dispositions:
                if not isinstance(disposition, Mapping):
                    raise LegacyCoverageError(f"invalid disposition: {receipt_path}")
                action_id = str(disposition.get("action_id") or "")
                if not action_id or action_id in disposition_by_action:
                    raise LegacyCoverageError(
                        f"missing or duplicate disposition action: {receipt_path}"
                    )
                disposition_by_action[action_id] = disposition

        for process_id in processes:
            if process_id < 0 or process_id >= len(action_ids):
                raise LegacyCoverageError(
                    f"process {cluster_id}.{process_id} is outside action map"
                )
            action_id = action_ids[process_id]
            run = run_map.get(action_id)
            if run is None:
                raise LegacyCoverageError(
                    f"receipt has no extracted run for {cluster_id}.{process_id} {action_id}"
                )
            relative_run = _relative_file(root, run, "legacy covered run")
            record = records_by_path.get(relative_run)
            if record is None:
                raise LegacyCoverageError(
                    f"scientific state does not contain {relative_run}"
                )
            payload = _read_object(run, "legacy covered run")
            if run.parent.name != action_id:
                raise LegacyCoverageError(f"run/action path mismatch: {run}")
            source_revision = str(
                _read_object(submission_path, "submission receipt").get(
                    "source_revision"
                )
                or ""
            )
            run_lock = payload.get("run_lock")
            run_revision = str(run_lock.get("code_revision") or "") if isinstance(run_lock, Mapping) else ""
            if source_revision and run_revision != source_revision:
                raise LegacyCoverageError(
                    f"run source revision mismatch: {cluster_id}.{process_id}"
                )
            disposition = disposition_by_action.get(action_id)
            disposition_status = "recorded"
            if disposition is not None:
                disposition_status = str(disposition.get("status") or "")
                if disposition_status not in _ACCEPTED_DISPOSITIONS:
                    raise LegacyCoverageError(
                        f"non-covering disposition for {cluster_id}.{process_id}"
                    )
                explicit_process = disposition.get("process_id")
                if explicit_process is not None and int(explicit_process) != process_id:
                    raise LegacyCoverageError(
                        f"disposition process mismatch for {cluster_id}.{process_id}"
                    )
                disposition_path = disposition.get("run_path")
                if disposition_path and Path(str(disposition_path)).as_posix() != relative_run:
                    raise LegacyCoverageError(
                        f"disposition run path mismatch for {cluster_id}.{process_id}"
                    )
            rows.append(
                {
                    "lane_id": lane_id,
                    "cluster_id": cluster_id,
                    "process_id": process_id,
                    "action_id": action_id,
                    "run_path": relative_run,
                    "run_sha256": _sha256_file(run),
                    "disposition_status": disposition_status,
                    "scientific_record_sha256": _canonical_sha256(record),
                    "receipt": _relative_file(root, receipt_path, "increment receipt"),
                    "receipt_sha256": _sha256_file(receipt_path),
                    **source,
                }
            )

    identities = [(row["cluster_id"], row["process_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise LegacyCoverageError("legacy receipts overlap process coverage")
    rows.sort(key=lambda row: (int(row["cluster_id"]), int(row["process_id"])))
    return {
        "schema": LEGACY_COVERAGE_SCHEMA_V1,
        "lane_id": lane_id,
        "recorded_at": recorded_at
        or _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(),
        "decision_handoff_sha256": decision_handoff_sha256,
        "scientific_state": _relative_file(root, ledger_path, "scientific state"),
        "normalized_receipt_count": len(
            {str(row["receipt"]) for row in rows}
        ),
        "covered_process_count": len(rows),
        "rows": rows,
    }


def write_legacy_coverage_index(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically persist one normalized coverage sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def validate_legacy_coverage_index(
    payload: Mapping[str, Any],
    *,
    repository_root: str | Path,
    lane_id: str,
    scientific_state: str | Path,
    decision_handoff_sha256: str,
) -> dict[tuple[int, int], dict[str, str]]:
    """Revalidate every sidecar binding against immutable source bytes."""

    if payload.get("schema") != LEGACY_COVERAGE_SCHEMA_V1:
        raise LegacyCoverageError("unsupported legacy coverage index schema")
    if payload.get("lane_id") != lane_id:
        raise LegacyCoverageError("legacy coverage lane mismatch")
    if payload.get("decision_handoff_sha256") != decision_handoff_sha256:
        raise LegacyCoverageError("legacy coverage decision epoch mismatch")
    root = Path(repository_root).resolve()
    ledger_path = Path(scientific_state).resolve()
    expected_ledger = _relative_file(root, ledger_path, "scientific state")
    if payload.get("scientific_state") != expected_ledger:
        raise LegacyCoverageError("legacy coverage scientific-state path mismatch")
    ledger = _read_object(ledger_path, "scientific state")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise LegacyCoverageError("scientific state has no records list")
    records_by_path = {
        str(record["run_path"]): record
        for record in records
        if isinstance(record, Mapping) and record.get("run_path")
    }
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise LegacyCoverageError("legacy coverage index has no rows")
    validated: dict[tuple[int, int], dict[str, str]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise LegacyCoverageError("legacy coverage row is not an object")
        cluster_id = int(raw["cluster_id"])
        process_id = int(raw["process_id"])
        key = (cluster_id, process_id)
        if key in validated:
            raise LegacyCoverageError(f"legacy coverage repeats {key}")
        if raw.get("lane_id") != lane_id:
            raise LegacyCoverageError(f"legacy row lane mismatch: {key}")
        if raw.get("disposition_status") not in _ACCEPTED_DISPOSITIONS:
            raise LegacyCoverageError(f"legacy row is not covering: {key}")
        for path_key, sha_key in (
            ("receipt", "receipt_sha256"),
            ("submission_receipt", "submission_receipt_sha256"),
            ("package_manifest", "package_manifest_sha256"),
            ("action_ids", "action_ids_sha256"),
            ("submit_descriptor", "submit_descriptor_sha256"),
            ("run_path", "run_sha256"),
        ):
            bound = root / str(raw[path_key])
            _relative_file(root, bound, path_key)
            if _sha256_file(bound) != raw.get(sha_key):
                raise LegacyCoverageError(f"legacy {path_key} digest mismatch: {key}")
        if raw.get("local_submission_audit"):
            audit_path = root / str(raw["local_submission_audit"])
            _relative_file(root, audit_path, "local submission audit")
            if _sha256_file(audit_path) != raw.get(
                "local_submission_audit_sha256"
            ):
                raise LegacyCoverageError(
                    f"legacy local submission audit digest mismatch: {key}"
                )
        receipt = _read_object(root / str(raw["receipt"]), "increment receipt")
        if receipt.get("schema") not in _ACCEPTED_RECEIPT_SCHEMAS or receipt.get(
            "status"
        ) not in _ACCEPTED_RECEIPT_STATUSES:
            raise LegacyCoverageError(f"legacy receipt no longer covers: {key}")
        processes = receipt.get("process_ids")
        if not isinstance(processes, list) or process_id not in {
            int(value) for value in processes
        }:
            raise LegacyCoverageError(f"legacy receipt process mismatch: {key}")
        submission = _read_object(
            root / str(raw["submission_receipt"]), "submission receipt"
        )
        if int(submission.get("cluster_id", -1)) != cluster_id:
            raise LegacyCoverageError(f"legacy submission cluster mismatch: {key}")
        action_ids = [
            line.strip()
            for line in (root / str(raw["action_ids"])).read_text().splitlines()
        ]
        if process_id >= len(action_ids) or action_ids[process_id] != raw.get(
            "action_id"
        ):
            raise LegacyCoverageError(f"legacy process/action mismatch: {key}")
        run_path = str(raw["run_path"])
        run = _read_object(root / run_path, "covered run")
        if (root / run_path).parent.name != raw.get("action_id"):
            raise LegacyCoverageError(f"legacy run/action mismatch: {key}")
        record = records_by_path.get(run_path)
        if record is None or _canonical_sha256(record) != raw.get(
            "scientific_record_sha256"
        ):
            raise LegacyCoverageError(f"legacy scientific-record mismatch: {key}")
        if not isinstance(run, Mapping):  # pragma: no cover - _read_object enforces
            raise LegacyCoverageError(f"legacy run is invalid: {key}")
        validated[key] = {
            "action_id": str(raw["action_id"]),
            "run_path": str((root / run_path).resolve()),
            "run_sha256": str(raw["run_sha256"]),
            "receipt": str((root / str(raw["receipt"])).resolve()),
        }
    if payload.get("covered_process_count") != len(validated):
        raise LegacyCoverageError("legacy coverage count mismatch")
    return validated


__all__ = [
    "LEGACY_COVERAGE_SCHEMA_V1",
    "LegacyCoverageError",
    "build_legacy_coverage_index",
    "validate_legacy_coverage_index",
    "write_legacy_coverage_index",
]
