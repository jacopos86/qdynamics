"""Materialize the explicit scheduler/frontier input for the combined report.

The combined report builder deliberately performs no scheduler or receipt
discovery.  This adapter turns one preserved, scoped scheduler snapshot plus a
digest-bound interpretation into the builder's narrow frontier contract.  It
also carries forward historical receipt indexes from the preceding canonical
report so a routine status refresh cannot silently drop provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pipelines.time_dynamics.diagnostics.combined_staged_campaign_report import (
    COMBINED_FRONTIER_SCHEMA,
)
from pipelines.time_dynamics.staged_campaign import CampaignLedger


SCHEDULER_SNAPSHOT_SCHEMA = "time_dynamics_scoped_scheduler_snapshot_v1"
FRONTIER_INTERPRETATION_SCHEMA = "paper_ii_report_frontier_interpretation_v1"


class FrontierMaterializationError(ValueError):
    """The preserved snapshot cannot support the requested report frontier."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FrontierMaterializationError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrontierMaterializationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FrontierMaterializationError(f"{label} must be a JSON object")
    return payload


def _relative(path: Path, *, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_timestamp(value: Any, *, label: str) -> str:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FrontierMaterializationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise FrontierMaterializationError(f"{label} must include a timezone")
    return text


def _snapshot_job_ids(snapshot: Mapping[str, Any]) -> set[str]:
    job_ids: set[str] = set()
    for index, raw in enumerate(snapshot.get("rows") or ()):
        row = dict(raw)
        try:
            cluster = int(row["cluster_id"])
            process = int(row["process_id"])
            status = int(row["job_status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FrontierMaterializationError(
                f"scheduler snapshot row {index} lacks integer identity/status"
            ) from exc
        if cluster <= 0 or process < 0 or status <= 0:
            raise FrontierMaterializationError(
                f"scheduler snapshot row {index} has invalid identity/status"
            )
        job_id = f"{cluster}.{process}"
        if job_id in job_ids:
            raise FrontierMaterializationError(
                f"scheduler snapshot contains duplicate job {job_id}"
            )
        job_ids.add(job_id)
    return job_ids


def materialize_report_frontier(
    *,
    snapshot_path: Path,
    interpretation_path: Path,
    ledger_path: Path,
    promotion_preview_path: Path,
    output_path: Path,
    prior_report_manifest_path: Path | None = None,
    workspace_root: Path = Path("."),
) -> dict[str, Any]:
    """Write one digest-bound combined-report frontier atomically."""

    workspace_root = workspace_root.resolve()
    snapshot_path = snapshot_path.resolve()
    interpretation_path = interpretation_path.resolve()
    ledger_path = ledger_path.resolve()
    promotion_preview_path = promotion_preview_path.resolve()
    output_path = output_path.resolve()

    snapshot = _read_object(snapshot_path, label="scheduler snapshot")
    interpretation = _read_object(
        interpretation_path, label="frontier interpretation"
    )
    if snapshot.get("schema") != SCHEDULER_SNAPSHOT_SCHEMA:
        raise FrontierMaterializationError("unexpected scheduler snapshot schema")
    if interpretation.get("schema") != FRONTIER_INTERPRETATION_SCHEMA:
        raise FrontierMaterializationError("unexpected frontier interpretation schema")
    snapshot_sha = _sha256(snapshot_path)
    if str(interpretation.get("scheduler_snapshot_sha256") or "") != snapshot_sha:
        raise FrontierMaterializationError(
            "frontier interpretation is stale against the scheduler snapshot"
        )
    snapshot_id = str(snapshot.get("snapshot_id") or "")
    if not snapshot_id or str(interpretation.get("snapshot_id") or "") != snapshot_id:
        raise FrontierMaterializationError("snapshot identity is absent or inconsistent")
    captured_at = _validate_timestamp(
        snapshot.get("captured_at_utc"), label="captured_at_utc"
    )
    queried = [int(value) for value in snapshot.get("queried_cluster_ids") or ()]
    if not queried or len(queried) != len(set(queried)):
        raise FrontierMaterializationError(
            "queried_cluster_ids must be a nonempty unique list"
        )
    observed_clusters = {
        int(dict(row)["cluster_id"]) for row in snapshot.get("rows") or ()
    }
    if not observed_clusters.issubset(set(queried)):
        raise FrontierMaterializationError(
            "scheduler snapshot contains a cluster outside its declared scope"
        )
    snapshot_job_ids = _snapshot_job_ids(snapshot)

    report_rows = [dict(row) for row in interpretation.get("scheduler_rows") or ()]
    for index, row in enumerate(report_rows):
        job_id = str(row.get("job_id") or "")
        if job_id not in snapshot_job_ids:
            raise FrontierMaterializationError(
                f"interpreted scheduler row {index} is absent from the snapshot: {job_id}"
            )

    ledger = CampaignLedger.read(ledger_path)
    preview = _read_object(promotion_preview_path, label="promotion preview")
    preview_ledger = dict(preview.get("ledger") or {})
    ledger_sha = _sha256(ledger_path)
    preview_sha = _sha256(promotion_preview_path)
    if (
        str(preview_ledger.get("sha256") or "") != ledger_sha
        or int(preview_ledger.get("record_count", -1)) != len(ledger.records)
    ):
        raise FrontierMaterializationError(
            "promotion preview is stale against the canonical ledger"
        )

    prior: dict[str, Any] = {}
    if prior_report_manifest_path is not None:
        prior_report_manifest_path = prior_report_manifest_path.resolve()
        prior = _read_object(prior_report_manifest_path, label="prior report manifest")
        if str(prior.get("source_ledger_sha256") or "") != ledger_sha:
            raise FrontierMaterializationError(
                "prior report manifest does not bind the current ledger"
            )

    incremental = list(
        interpretation.get("incremental_ingestion_receipts")
        if "incremental_ingestion_receipts" in interpretation
        else prior.get("incremental_ingestion_receipts") or ()
    )
    submissions = list(
        interpretation.get("submission_receipts")
        if "submission_receipts" in interpretation
        else prior.get("submission_receipts") or ()
    )
    historical_clusters = [
        int(value)
        for value in (
            interpretation.get("cluster_ids")
            if "cluster_ids" in interpretation
            else prior.get("cluster_ids") or ()
        )
    ]
    paper_clusters = [
        int(value) for value in interpretation.get("paper_cluster_ids") or ()
    ]
    if not set(paper_clusters).issubset(set(queried)):
        raise FrontierMaterializationError(
            "paper_cluster_ids are not contained in the scoped scheduler snapshot"
        )
    cluster_ids = sorted(set(historical_clusters).union(paper_clusters))

    frontier = {
        "schema": COMBINED_FRONTIER_SCHEMA,
        "snapshot_id": snapshot_id,
        "snapshot_utc": captured_at,
        "audit_status": str(interpretation.get("audit_status") or ""),
        "scheduler_summary": str(interpretation.get("scheduler_summary") or ""),
        "scheduler_snapshot": {
            "path": _relative(snapshot_path, workspace_root=workspace_root),
            "sha256": snapshot_sha,
            "queried_cluster_ids": queried,
        },
        "frontier_interpretation": {
            "path": _relative(interpretation_path, workspace_root=workspace_root),
            "sha256": _sha256(interpretation_path),
        },
        "ledger": {"sha256": ledger_sha, "record_count": len(ledger.records)},
        "promotion_preview": {
            "sha256": preview_sha,
            "planned_tuple_count": int(preview.get("planned_tuple_count", -1)),
        },
        "cluster_ids": cluster_ids,
        "active_clusters": list(interpretation.get("active_clusters") or ()),
        "excluded_cluster_ids": [
            int(value) for value in interpretation.get("excluded_cluster_ids") or ()
        ],
        "incremental_ingestion_receipts": incremental,
        "submission_receipts": submissions,
        "scheduler_rows": report_rows,
        "supplemental_sections": list(
            interpretation.get("supplemental_sections") or ()
        ),
    }
    if not frontier["audit_status"] or not frontier["scheduler_summary"]:
        raise FrontierMaterializationError(
            "frontier interpretation requires audit_status and scheduler_summary"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    if temporary.exists():
        raise FrontierMaterializationError(
            f"stale frontier temporary file requires review: {temporary}"
        )
    temporary.write_text(
        json.dumps(frontier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--promotion-preview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-report-manifest", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args()
    materialize_report_frontier(
        snapshot_path=args.snapshot,
        interpretation_path=args.interpretation,
        ledger_path=args.ledger,
        promotion_preview_path=args.promotion_preview,
        output_path=args.output,
        prior_report_manifest_path=args.prior_report_manifest,
        workspace_root=args.workspace_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FRONTIER_INTERPRETATION_SCHEMA",
    "FrontierMaterializationError",
    "SCHEDULER_SNAPSHOT_SCHEMA",
    "materialize_report_frontier",
]
