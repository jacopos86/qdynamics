"""Legacy receipt normalization proves process/action/run identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.legacy_coverage import (
    LegacyCoverageError,
    build_legacy_coverage_index,
    validate_legacy_coverage_index,
)
from pipelines.time_dynamics.acquisition_supervisor import (
    LaneRegistration,
    SchedulerJob,
    _validated_coverage_for_jobs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path
    package = root / "chtc/frontier"
    input_root = package / "input"
    input_root.mkdir(parents=True)
    action_id = "action-0"
    action = {"action_id": action_id, "runner_argv": ["--code-revision", "abc"]}
    manifest = input_root / "manifest.json"
    manifest.write_text(json.dumps({"actions": [action]}) + "\n")
    action_ids = input_root / "action_ids.txt"
    action_ids.write_text(action_id + "\n")
    descriptor = package / "submit.sub"
    descriptor.write_text(
        "queue action_id from chtc/frontier/input/action_ids.txt\n"
    )
    submission = package / "submission_receipt.json"
    submission.write_text(
        json.dumps(
            {
                "cluster_id": 42,
                "status": "SUBMITTED",
                "submitted_jobs": 1,
                "source_revision": "abc",
                "package": {
                    "manifest_sha256": _sha(manifest),
                    "submit_descriptor_sha256": _sha(descriptor),
                },
            }
        )
        + "\n"
    )
    increment = root / "output/attempts/cluster_42/increments/proc0"
    evidence = increment / "evidence"
    run = increment / f"extracted/raw/{action_id}/run.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"run_lock": {"code_revision": "abc"}}) + "\n")
    evidence.mkdir(parents=True)
    receipt = evidence / "increment_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "paper_ii_incremental_ingestion_receipt_v1",
                "status": "INGESTED_AND_REPORTED",
                "cluster_id": 42,
                "process_ids": [0],
            }
        )
        + "\n"
    )
    relative_run = run.relative_to(root).as_posix()
    ledger = root / "output/ledger.json"
    ledger.write_text(json.dumps({"records": [{"run_path": relative_run}]}) + "\n")
    return {
        "root": root,
        "attempts": root / "output/attempts",
        "chtc": root / "chtc",
        "ledger": ledger,
        "run": run,
        "receipt": receipt,
    }


def _build(paths: dict[str, Path]) -> dict:
    return build_legacy_coverage_index(
        repository_root=paths["root"],
        lane_id="paper",
        attempts_root=paths["attempts"],
        chtc_root=paths["chtc"],
        scientific_state=paths["ledger"],
        decision_handoff_sha256="d" * 64,
        recorded_at="2026-08-28T00:00:00Z",
    )


def test_build_and_validate_strong_legacy_coverage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = _build(paths)
    assert index["covered_process_count"] == 1
    row = index["rows"][0]
    assert (row["cluster_id"], row["process_id"], row["action_id"]) == (
        42,
        0,
        "action-0",
    )
    assert row["run_sha256"] == _sha(paths["run"])
    validated = validate_legacy_coverage_index(
        index,
        repository_root=paths["root"],
        lane_id="paper",
        scientific_state=paths["ledger"],
        decision_handoff_sha256="d" * 64,
    )
    assert validated[(42, 0)]["action_id"] == "action-0"


def test_digest_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = _build(paths)
    paths["run"].write_text('{"tampered":true}\n')
    with pytest.raises(LegacyCoverageError, match="run_path digest mismatch"):
        validate_legacy_coverage_index(
            index,
            repository_root=paths["root"],
            lane_id="paper",
            scientific_state=paths["ledger"],
            decision_handoff_sha256="d" * 64,
        )


def test_noncanonical_diagnostic_receipt_is_not_normalized(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text())
    receipt["schema"] = "paper_ii_incremental_diagnostic_receipt_v1"
    receipt["status"] = "VALIDATED_DIAGNOSTIC_NONCANONICAL_REISSUE_REQUIRED"
    paths["receipt"].write_text(json.dumps(receipt) + "\n")
    index = _build(paths)
    assert index["covered_process_count"] == 0


def test_supervisor_uses_sidecar_without_weakening_legacy_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = _build(paths)
    index_path = paths["root"] / "output/legacy_index.json"
    index_path.write_text(json.dumps(index) + "\n")
    lane = LaneRegistration(
        lane_id="paper",
        kind="paper",
        monitor_enabled=True,
        submission_enabled=False,
        attempts_root=paths["attempts"],
        state_root=paths["root"] / "output/state",
        scientific_state=paths["ledger"],
        metadata={
            "legacy_coverage_index": index_path.relative_to(paths["root"]).as_posix(),
            "decision_handoff_sha256": "d" * 64,
        },
    )
    job = SchedulerJob(
        lane_id="paper",
        cluster_id=42,
        process_id=0,
        state="succeeded",
        action_id="action-0",
        exit_code=0,
    )
    coverage = _validated_coverage_for_jobs(
        repository_root=paths["root"], lane=lane, jobs=(job,)
    )
    assert coverage[job.key]["run_sha256"] == _sha(paths["run"])
