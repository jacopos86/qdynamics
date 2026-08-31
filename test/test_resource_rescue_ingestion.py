from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 import (
    CAMPAIGN_ID,
    build_campaign,
)
from pipelines.time_dynamics.resource_rescue_ingestion import (
    load_resource_rescue_context,
)
from pipelines.time_dynamics.staged_campaign import CampaignLedger, CellAction


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path: Path) -> tuple[Path, str]:
    package = tmp_path / "rescue"
    input_dir = package / "input"
    campaign = build_campaign(prior_root=None)
    snapshot = CampaignLedger(
        qualification_policy=campaign.accuracy_policy.to_json_dict()
    )
    snapshot_path = input_dir / "ledger_snapshot.json"
    snapshot.write(snapshot_path)
    action = CellAction(
        key=campaign.cells[0],
        stage="screen_avqds_t5",
        action="run",
        horizon="t5",
        threshold=9.0e-6,
        reason="resource rescue",
    )
    row = {
        "action_id": action.action_id,
        "action": action.to_json_dict(),
        "packaged_action": action.to_json_dict(),
        "trajectory_json": f"raw/{action.action_id}/run.json",
    }
    actions_path = input_dir / "actions.jsonl"
    actions_path.write_text(json.dumps(row, sort_keys=True) + "\n")
    plan = {
        "schema": "paper_ii_resource_rescue_plan_v1",
        "status": "PREPARED_NOT_SUBMITTED",
        "source_revision": "a" * 40,
        "prior_root": {"state": "absent"},
        "canonical_ledger": {
            "record_count": 0,
            "sha256": _sha(snapshot_path),
        },
        "selected_action_ids": [action.action_id],
    }
    plan_path = input_dir / "resource_rescue_plan.json"
    _write_json(plan_path, plan)
    manifest = {
        "schema": "paper_ii_staged_chtc_package_v1",
        "campaign_id": CAMPAIGN_ID,
        "stage": "resource_rescue_t5_frontier",
        "status": "PREPARED_NOT_SUBMITTED",
        "code_revision": "a" * 40,
        "action_count": 1,
        "actions": [row],
        "resource_rescue_plan": {"sha256": _sha(plan_path)},
        "ledger_snapshot": {"sha256": _sha(snapshot_path)},
    }
    _write_json(input_dir / "manifest.json", manifest)
    return package, action.action_id


def test_load_resource_rescue_context_binds_plan_manifest_and_snapshot(
    tmp_path: Path,
) -> None:
    package, action_id = _package(tmp_path)
    context = load_resource_rescue_context(package)
    assert context.plan["selected_action_ids"] == [action_id]
    assert context.action_rows[0]["action_id"] == action_id
    assert context.ledger_snapshot.records == ()


def test_load_resource_rescue_context_rejects_unplanned_action(tmp_path: Path) -> None:
    package, _action_id = _package(tmp_path)
    plan_path = package / "input/resource_rescue_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["selected_action_ids"] = ["not-the-packaged-action"]
    _write_json(plan_path, plan)
    manifest_path = package / "input/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resource_rescue_plan"]["sha256"] = _sha(plan_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identities differ"):
        load_resource_rescue_context(package)


def test_load_resource_rescue_context_rejects_changed_snapshot(tmp_path: Path) -> None:
    package, _action_id = _package(tmp_path)
    snapshot_path = package / "input/ledger_snapshot.json"
    snapshot_path.write_text(snapshot_path.read_text() + "\n")
    with pytest.raises(ValueError, match="snapshot digest"):
        load_resource_rescue_context(package)
