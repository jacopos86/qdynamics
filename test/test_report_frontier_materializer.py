"""Explicit scheduler-snapshot to combined-report frontier contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.diagnostics.report_frontier_materializer import (
    FRONTIER_INTERPRETATION_SCHEMA,
    FrontierMaterializationError,
    SCHEDULER_SNAPSHOT_SCHEMA,
    materialize_report_frontier,
)
from pipelines.time_dynamics.staged_campaign import CampaignLedger


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case(tmp_path: Path) -> dict[str, Path]:
    ledger = tmp_path / "output/ledger.json"
    CampaignLedger().write(ledger)
    preview = tmp_path / "output/preview.json"
    _write(
        preview,
        {
            "ledger": {
                "sha256": _sha256(ledger),
                "record_count": 0,
            },
            "planned_tuple_count": 18,
        },
    )
    snapshot = tmp_path / "output/snapshot.json"
    _write(
        snapshot,
        {
            "schema": SCHEDULER_SNAPSHOT_SCHEMA,
            "snapshot_id": "snapshot-1",
            "captured_at_utc": "2026-08-28T20:13:30.991Z",
            "queried_cluster_ids": [11, 12],
            "rows": [
                {
                    "cluster_id": 11,
                    "process_id": 3,
                    "job_status": 4,
                    "exit_code": 1,
                    "hold_reason": None,
                    "job_batch_name": "paper",
                }
            ],
        },
    )
    interpretation = tmp_path / "output/interpretation.json"
    _write(
        interpretation,
        {
            "schema": FRONTIER_INTERPRETATION_SCHEMA,
            "snapshot_id": "snapshot-1",
            "scheduler_snapshot_sha256": _sha256(snapshot),
            "paper_cluster_ids": [11],
            "audit_status": "PASS",
            "scheduler_summary": "one terminal transfer-pending delta",
            "active_clusters": [{"cluster_id": 11, "status": "transfer_pending"}],
            "scheduler_rows": [
                {
                    "job_id": "11.3",
                    "method": "AVQDS",
                    "controller": "state motion",
                    "drive": "strongfast",
                    "threshold": 9.0e-6,
                    "state": "transfer_pending",
                    "progress": "exit 1",
                    "available_data": "remote archive verified",
                }
            ],
        },
    )
    prior = tmp_path / "output/report.json"
    _write(
        prior,
        {
            "source_ledger_sha256": _sha256(ledger),
            "cluster_ids": [1, 2],
            "incremental_ingestion_receipts": ["increment.json"],
            "submission_receipts": ["submission.json"],
        },
    )
    return {
        "ledger": ledger,
        "preview": preview,
        "snapshot": snapshot,
        "interpretation": interpretation,
        "prior": prior,
        "output": tmp_path / "output/frontier.json",
    }


def test_materializer_binds_snapshot_and_preserves_prior_indexes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    frontier = materialize_report_frontier(
        snapshot_path=case["snapshot"],
        interpretation_path=case["interpretation"],
        ledger_path=case["ledger"],
        promotion_preview_path=case["preview"],
        output_path=case["output"],
        prior_report_manifest_path=case["prior"],
        workspace_root=tmp_path,
    )

    assert frontier["cluster_ids"] == [1, 2, 11]
    assert frontier["scheduler_snapshot"]["sha256"] == _sha256(case["snapshot"])
    assert frontier["incremental_ingestion_receipts"] == ["increment.json"]
    assert frontier["submission_receipts"] == ["submission.json"]
    assert case["output"].is_file()
    assert not case["output"].with_name("frontier.json.part").exists()


def test_materializer_rejects_interpretation_for_another_snapshot(tmp_path: Path) -> None:
    case = _case(tmp_path)
    interpretation = json.loads(case["interpretation"].read_text())
    interpretation["scheduler_snapshot_sha256"] = "0" * 64
    _write(case["interpretation"], interpretation)

    with pytest.raises(FrontierMaterializationError, match="stale"):
        materialize_report_frontier(
            snapshot_path=case["snapshot"],
            interpretation_path=case["interpretation"],
            ledger_path=case["ledger"],
            promotion_preview_path=case["preview"],
            output_path=case["output"],
            workspace_root=tmp_path,
        )


def test_materializer_rejects_report_row_absent_from_scoped_snapshot(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    interpretation = json.loads(case["interpretation"].read_text())
    interpretation["scheduler_rows"][0]["job_id"] = "11.4"
    _write(case["interpretation"], interpretation)

    with pytest.raises(FrontierMaterializationError, match="absent"):
        materialize_report_frontier(
            snapshot_path=case["snapshot"],
            interpretation_path=case["interpretation"],
            ledger_path=case["ledger"],
            promotion_preview_path=case["preview"],
            output_path=case["output"],
            workspace_root=tmp_path,
        )
