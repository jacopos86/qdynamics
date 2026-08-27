#!/usr/bin/env python3
"""Deterministic local action packages for the Paper III static-QSE campaign.

A package is a directory of self-describing action files plus one command
manifest. This is **packaging, not remote submission**: every command runs
the local one-action producer

```text
python -m pipelines.exact_bench.paper_iii_static_qse_cell \
  --action-json <package_root>/<action_id>/ACTION.json \
  --output-dir  <package_root>/<action_id>
```

so each action's outputs land next to its action file and the package root
doubles as the artifact root for exact-subset ingestion. Package order and
commands are deterministic (campaign planning order), independent of any
directory listing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipelines.qse_spectra.static_campaign import (
    StaticCampaignError,
    StaticQSEAction,
    StaticQSECampaign,
    canonical_digest,
)

PACKAGE_SCHEMA_VERSION = "paper_iii_static_qse_package_v1"
ACTION_FILE_SCHEMA_VERSION = "paper_iii_static_qse_action_file_v1"
ACTION_FILENAME = "ACTION.json"
PACKAGE_MANIFEST_FILENAME = "package_manifest.json"


@dataclass(frozen=True)
class StaticQSEPackage:
    """One built local package: ordered actions plus their execution root."""

    package_root: Path
    campaign_digest: str
    actions: tuple[StaticQSEAction, ...]

    @property
    def artifact_root(self) -> Path:
        return Path(self.package_root)

    @property
    def actions_by_id(self) -> dict[str, StaticQSEAction]:
        return {action.action_id: action for action in self.actions}

    def commands(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            build_action_command(action, self.action_dir(action))
            for action in self.actions
        )

    def action_dir(self, action: StaticQSEAction) -> Path:
        return Path(self.package_root) / action.action_id


def build_action_command(action: StaticQSEAction, output_dir: Path) -> tuple[str, ...]:
    """The deterministic local producer invocation for one action."""

    out = Path(output_dir)
    return (
        "python",
        "-m",
        "pipelines.exact_bench.paper_iii_static_qse_cell",
        "--action-json",
        str(out / ACTION_FILENAME),
        "--output-dir",
        str(out),
    )


def _validate_context(action: StaticQSEAction, context: Mapping[str, Any]) -> None:
    spectral = context.get("spectral_request")
    if not isinstance(spectral, Mapping):
        raise StaticCampaignError(
            f"action {action.action_id}: context requires a spectral_request mapping."
        )
    declared = str(action.cell.spectral_request_digest)
    actual = canonical_digest(dict(spectral))
    if actual != declared:
        raise StaticCampaignError(
            f"action {action.action_id}: spectral_request digest {actual} does not "
            f"match the cell's declared {declared}."
        )
    problem = context.get("problem")
    if not isinstance(problem, Mapping) or "kind" not in problem:
        raise StaticCampaignError(
            f"action {action.action_id}: context requires a problem mapping with kind."
        )
    ground = context.get("ground_state")
    if not isinstance(ground, Mapping) or "kind" not in ground:
        raise StaticCampaignError(
            f"action {action.action_id}: context requires a ground_state mapping with kind."
        )


def build_local_package(
    campaign: StaticQSECampaign,
    actions: Sequence[StaticQSEAction],
    *,
    package_root: Path,
    cell_contexts: Mapping[str, Mapping[str, Any]],
) -> StaticQSEPackage:
    """Write a deterministic local package for the given planned actions.

    ``cell_contexts`` is keyed by cell digest and supplies the execution
    context (problem declaration, ground-state source, trace and spectral
    requests, qualification policy). Contexts must satisfy the cell's
    declared spectral-request digest.
    """

    root = Path(package_root)
    root.mkdir(parents=True, exist_ok=True)
    declared_ids = {a.action_id for a in campaign.all_actions()}
    ordered = sorted(
        (a for a in actions),
        key=lambda a: (a.cell.digest(), a.stage, a.action_id),
    )
    if not ordered:
        raise StaticCampaignError("a package requires at least one action.")
    entries: list[dict[str, Any]] = []
    for action in ordered:
        if action.action_id not in declared_ids:
            raise StaticCampaignError(
                f"action {action.action_id} is not declared by campaign "
                f"{campaign.campaign_id!r}."
            )
        cell_digest = action.cell.digest()
        context = cell_contexts.get(cell_digest)
        if context is None:
            raise StaticCampaignError(
                f"missing execution context for cell {cell_digest}."
            )
        _validate_context(action, context)
        action_dir = root / action.action_id
        action_dir.mkdir(parents=True, exist_ok=True)
        action_file = {
            "schema_version": ACTION_FILE_SCHEMA_VERSION,
            "action_id": action.action_id,
            "action": action.payload(),
            "context": dict(context),
        }
        (action_dir / ACTION_FILENAME).write_text(
            json.dumps(action_file, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        entries.append(
            {
                "action_id": action.action_id,
                "stage": str(action.stage),
                "cell_digest": cell_digest,
                "method_id": str(action.cell.method_id),
                "case_id": str(action.cell.physics.case_id),
                "dependency_action_ids": list(action.dependency_action_ids),
                "command": list(build_action_command(action, action_dir)),
                "context_digest": canonical_digest(dict(context)),
            }
        )
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "campaign_id": str(campaign.campaign_id),
        "campaign_digest": campaign.digest(),
        "action_count": len(entries),
        "actions": entries,
    }
    manifest["manifest_digest"] = canonical_digest(
        {k: v for k, v in manifest.items() if k != "manifest_digest"}
    )
    (root / PACKAGE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return StaticQSEPackage(
        package_root=root,
        campaign_digest=campaign.digest(),
        actions=tuple(ordered),
    )


def load_local_package(
    campaign: StaticQSECampaign, package_root: Path
) -> StaticQSEPackage:
    """Reload a package and re-verify it against the declared campaign."""

    root = Path(package_root)
    manifest_path = root / PACKAGE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise StaticCampaignError(f"missing package manifest at {manifest_path}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise StaticCampaignError(
            f"package schema mismatch: {manifest.get('schema_version')!r}."
        )
    if str(manifest.get("campaign_digest")) != campaign.digest():
        raise StaticCampaignError(
            "package campaign digest does not match the declared campaign."
        )
    expected_digest = canonical_digest(
        {k: v for k, v in manifest.items() if k != "manifest_digest"}
    )
    if str(manifest.get("manifest_digest")) != expected_digest:
        raise StaticCampaignError("package manifest digest drift.")
    by_id = {a.action_id: a for a in campaign.all_actions()}
    actions: list[StaticQSEAction] = []
    for entry in manifest.get("actions", []):
        action_id = str(entry.get("action_id"))
        action = by_id.get(action_id)
        if action is None:
            raise StaticCampaignError(
                f"package lists action {action_id} not declared by the campaign."
            )
        actions.append(action)
    return StaticQSEPackage(
        package_root=root,
        campaign_digest=campaign.digest(),
        actions=tuple(actions),
    )


__all__ = [
    "ACTION_FILENAME",
    "ACTION_FILE_SCHEMA_VERSION",
    "PACKAGE_MANIFEST_FILENAME",
    "PACKAGE_SCHEMA_VERSION",
    "StaticQSEPackage",
    "build_action_command",
    "build_local_package",
    "load_local_package",
]
