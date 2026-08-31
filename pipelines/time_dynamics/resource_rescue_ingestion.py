"""Ingest explicitly planned Paper-II resource-rescue outcomes.

Ordinary staged packages must remain members of ``StagedCampaign.next_actions``
at ingestion time.  A resource-rescue package is different: it deliberately
tests a coarser activation rung after a controller-resource safeguard.  This
module preserves the normal stale-action rejection and admits such evidence
only through the immutable rescue plan that authorized it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 import (
    CAMPAIGN_ID,
    build_campaign,
)
from pipelines.time_dynamics.staged_campaign import CampaignLedger, CellAction
from pipelines.time_dynamics.staged_campaign_artifacts import record_run_outcome


RESOURCE_RESCUE_PLAN_SCHEMA = "paper_ii_resource_rescue_plan_v1"
RESOURCE_RESCUE_STAGE = "resource_rescue_t5_frontier"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("resource-rescue actions JSONL is empty or malformed")
    return tuple(dict(row) for row in rows)


@dataclass(frozen=True)
class ResourceRescueContext:
    """Verified immutable control plane for one rescue package."""

    manifest: Mapping[str, Any]
    plan: Mapping[str, Any]
    ledger_snapshot: CampaignLedger
    action_rows: tuple[dict[str, Any], ...]


def load_resource_rescue_context(package_dir: str | Path) -> ResourceRescueContext:
    """Verify one resource-rescue package without consulting run outputs."""

    package = Path(package_dir)
    input_dir = package / "input"
    manifest_path = input_dir / "manifest.json"
    plan_path = input_dir / "resource_rescue_plan.json"
    snapshot_path = input_dir / "ledger_snapshot.json"
    actions_path = input_dir / "actions.jsonl"

    manifest = _read_object(manifest_path, label="resource-rescue manifest")
    plan = _read_object(plan_path, label="resource-rescue plan")
    if manifest.get("schema") != "paper_ii_staged_chtc_package_v1":
        raise ValueError("resource-rescue package has an unsupported manifest schema")
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("resource-rescue package names the wrong campaign")
    if manifest.get("stage") != RESOURCE_RESCUE_STAGE:
        raise ValueError("package is not an explicit resource-rescue stage")
    if plan.get("schema") != RESOURCE_RESCUE_PLAN_SCHEMA:
        raise ValueError("resource-rescue plan has an unsupported schema")
    if plan.get("status") not in {"PREPARED_NOT_SUBMITTED", "SUBMITTED"}:
        raise ValueError("resource-rescue plan is not in an ingestible lifecycle state")
    if dict(plan.get("prior_root") or {}) != {"state": "absent"}:
        raise ValueError("resource-rescue ingestion requires absent prior_root provenance")
    if manifest.get("code_revision") != plan.get("source_revision"):
        raise ValueError("resource-rescue source revision differs across plan and manifest")

    manifest_plan = dict(manifest.get("resource_rescue_plan") or {})
    manifest_snapshot = dict(manifest.get("ledger_snapshot") or {})
    if manifest_plan.get("sha256") != _sha256(plan_path):
        raise ValueError("resource-rescue plan digest differs from the manifest")
    if manifest_snapshot.get("sha256") != _sha256(snapshot_path):
        raise ValueError("resource-rescue ledger snapshot digest differs from the manifest")

    canonical = dict(plan.get("canonical_ledger") or {})
    if canonical.get("sha256") != _sha256(snapshot_path):
        raise ValueError("resource-rescue plan does not bind the packaged ledger snapshot")
    snapshot = CampaignLedger.read(snapshot_path)
    if canonical.get("record_count") != len(snapshot.records):
        raise ValueError("resource-rescue plan ledger count is inconsistent")

    rows = _read_jsonl(actions_path)
    if list(rows) != list(manifest.get("actions") or ()):
        raise ValueError("actions JSONL differs from the manifest action rows")
    row_ids = tuple(str(row.get("action_id") or "") for row in rows)
    selected_ids = tuple(str(value) for value in plan.get("selected_action_ids") or ())
    if (
        not all(row_ids)
        or len(set(row_ids)) != len(row_ids)
        or row_ids != selected_ids
        or len(rows) != int(manifest.get("action_count") or -1)
    ):
        raise ValueError("resource-rescue action identities differ across control files")

    for row in rows:
        action = CellAction.from_json_dict(dict(row.get("action") or {}))
        if row["action_id"] != action.action_id:
            raise ValueError("resource-rescue row identity differs from its action")
        if row.get("packaged_action") != row.get("action"):
            raise ValueError("resource-rescue packaged action differs from its source action")
        if action.horizon != "t5" or action.resume_from_run_path is not None:
            raise ValueError("resource-rescue actions must be fresh t5 screens")
        if action.threshold != 9.0e-6:
            raise ValueError("resource-rescue action uses an unauthorized activation cut")
        trajectory = Path(str(row.get("trajectory_json") or ""))
        if trajectory.is_absolute() or ".." in trajectory.parts:
            raise ValueError("resource-rescue trajectory escapes its extraction root")

    return ResourceRescueContext(
        manifest=manifest,
        plan=plan,
        ledger_snapshot=snapshot,
        action_rows=rows,
    )


def record_resource_rescue_outputs(
    *,
    package_dir: str | Path,
    ledger: CampaignLedger,
    artifact_root: str | Path,
    seed_path: str | Path,
    action_ids: Sequence[str] = (),
    reference_substeps: int = 40,
    reference_store_root: str | Path = "output/paper_ii_staged/exact_references",
) -> tuple[CampaignLedger, tuple[dict[str, Any], ...]]:
    """Validate and append an exact subset of a planned rescue package."""

    context = load_resource_rescue_context(package_dir)
    snapshot = context.ledger_snapshot
    prefix_count = len(snapshot.records)
    if len(ledger.records) < prefix_count or (
        ledger.records[:prefix_count] != snapshot.records
        or dict(ledger.qualification_policy) != dict(snapshot.qualification_policy)
    ):
        raise ValueError("live ledger no longer contains the frozen rescue-plan prefix")

    selected = tuple(str(value) for value in action_ids)
    if len(set(selected)) != len(selected):
        raise ValueError("resource-rescue action selection contains duplicates")
    authorized = {str(row["action_id"]): row for row in context.action_rows}
    unknown = sorted(set(selected) - set(authorized))
    if unknown:
        raise ValueError(f"unplanned resource-rescue action selection: {unknown}")
    rows = (
        tuple(authorized[action_id] for action_id in selected)
        if selected
        else context.action_rows
    )

    root = Path(artifact_root)
    if not root.is_dir():
        raise ValueError(f"resource-rescue extraction root does not exist: {root}")
    campaign = build_campaign(prior_root=None)
    if dict(ledger.qualification_policy) != campaign.accuracy_policy.to_json_dict():
        raise ValueError("live ledger is not classified under the canonical policy")

    current = ledger
    receipts: list[dict[str, Any]] = []
    for row in rows:
        action = CellAction.from_json_dict(dict(row["action"]))
        run_path = root / Path(str(row["trajectory_json"]))
        existing = current.for_cell(action.key, action.horizon)
        duplicate = next(
            (record for record in existing if record.threshold == action.threshold),
            None,
        )
        if duplicate is not None:
            if Path(str(duplicate.run_path or "")) != run_path:
                raise ValueError(
                    "live ledger already has different evidence for a planned rescue rung"
                )
            receipts.append(
                {
                    "action_id": action.action_id,
                    "run_path": str(run_path),
                    "status": "already_recorded",
                    "ledger_status": duplicate.status,
                }
            )
            continue
        if not run_path.is_file() or run_path.stat().st_size <= 0:
            raise ValueError(f"resource-rescue run is missing or empty: {run_path}")
        command = campaign.build_run_command(
            action,
            seed_path=str(seed_path),
            output_json=str(run_path),
            numerics=str(context.manifest["numerics"]),
            gate=str(context.manifest["gate"]),
            search_profile=str(context.manifest["search_profile"]),
        )
        current = record_run_outcome(
            campaign=campaign,
            ledger=current,
            action=action,
            command=command,
            run_path=run_path,
            seed_path=seed_path,
            reference_substeps=int(reference_substeps),
            reference_store_root=reference_store_root,
        )
        record = current.records[-1]
        receipts.append(
            {
                "action_id": action.action_id,
                "run_path": str(run_path),
                "status": "recorded",
                "ledger_status": record.status,
            }
        )
    return current, tuple(receipts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed-json", type=Path, required=True)
    parser.add_argument("--action-id", action="append", default=[])
    parser.add_argument("--reference-substeps", type=int, default=40)
    parser.add_argument(
        "--reference-store-root",
        type=Path,
        default=Path("output/paper_ii_staged/exact_references"),
    )
    args = parser.parse_args(argv)
    ledger = CampaignLedger.read(args.ledger)
    updated, receipts = record_resource_rescue_outputs(
        package_dir=args.package_dir,
        ledger=ledger,
        artifact_root=args.artifact_root,
        seed_path=args.seed_json,
        action_ids=tuple(args.action_id),
        reference_substeps=args.reference_substeps,
        reference_store_root=args.reference_store_root,
    )
    updated.write(args.ledger)
    print(
        json.dumps(
            {
                "ledger": str(args.ledger),
                "ledger_record_count": len(updated.records),
                "receipts": list(receipts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RESOURCE_RESCUE_PLAN_SCHEMA",
    "RESOURCE_RESCUE_STAGE",
    "ResourceRescueContext",
    "load_resource_rescue_context",
    "record_resource_rescue_outputs",
]
