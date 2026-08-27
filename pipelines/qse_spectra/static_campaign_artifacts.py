#!/usr/bin/env python3
"""Verification and ingestion of Paper III static-QSE action outputs.

One completed scheduler process is independently ingestible evidence: any
exact nonempty subset of a package's action ids can be admitted as soon as
its own artifacts exist. Admission is achieved-configuration-validated and
idempotent; unknown ids, unverified outputs, achieved-setting mismatches,
and conflicting replays fail closed. A completed numerical result that
fails a gate is admitted as ``candidate`` with its values intact -- only
absent work is pending.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping

from pipelines.qse_spectra.static_campaign import (
    AdmissionResult,
    StaticCampaignError,
    StaticQSEAction,
    StaticQSEAttempt,
    StaticQSELedger,
    normalize_method_id,
)

RESULT_ENVELOPE_SCHEMA_VERSION = "paper_iii_static_qse_result_envelope_v1"
RESULT_ENVELOPE_FILENAME = "result_envelope.json"


class StaticCampaignArtifactError(StaticCampaignError):
    """Raised when an action's artifacts fail verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_output_dir(artifact_root: Path, action_id: str) -> Path:
    return Path(artifact_root) / str(action_id)


def build_result_envelope(
    *,
    action: StaticQSEAction,
    achieved_settings: Mapping[str, Any],
    source_revision: str,
    input_digests: Mapping[str, str],
    payload_files: Mapping[str, Path],
    disposition: str,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The self-describing result envelope one producer process writes.

    ``payload_files`` maps envelope-relative names to already-written files
    in the action's output directory; their hashes become the output digests.
    """

    output_digests = {
        str(name): sha256_file(Path(path)) for name, path in sorted(payload_files.items())
    }
    achieved = {
        "cell_digest": action.cell.digest(),
        "stage": str(action.stage),
        "method_id": normalize_method_id(action.cell.method_id),
        "physics_digest": action.cell.physics.digest(),
        "settings": dict(achieved_settings),
    }
    return {
        "schema_version": RESULT_ENVELOPE_SCHEMA_VERSION,
        "action_id": action.action_id,
        "action": action.payload(),
        "achieved": achieved,
        "source_revision": str(source_revision),
        "input_digests": {str(k): str(v) for k, v in sorted(input_digests.items())},
        "payload_files": sorted(str(name) for name in payload_files),
        "output_digests": output_digests,
        "disposition": str(disposition),
        "telemetry": dict(telemetry or {}),
    }


def write_result_envelope(envelope: Mapping[str, Any], output_dir: Path) -> Path:
    path = Path(output_dir) / RESULT_ENVELOPE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(envelope), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def verify_achieved_action(
    action: StaticQSEAction, artifact_root: Path
) -> StaticQSEAttempt:
    """Re-read one action's artifacts and return the verified attempt.

    Fails closed on a missing envelope, action-id or achieved-configuration
    mismatch (method aliases are normalized before comparison), missing
    payload files, or output-digest drift.
    """

    out_dir = action_output_dir(artifact_root, action.action_id)
    envelope_path = out_dir / RESULT_ENVELOPE_FILENAME
    if not envelope_path.is_file():
        raise StaticCampaignArtifactError(
            f"missing result envelope for action {action.action_id} at {envelope_path}."
        )
    try:
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticCampaignArtifactError(
            f"unreadable result envelope at {envelope_path}: {exc}"
        ) from exc
    if envelope.get("schema_version") != RESULT_ENVELOPE_SCHEMA_VERSION:
        raise StaticCampaignArtifactError(
            f"result envelope schema mismatch at {envelope_path}: "
            f"{envelope.get('schema_version')!r}"
        )
    if str(envelope.get("action_id")) != action.action_id:
        raise StaticCampaignArtifactError(
            f"envelope action id {envelope.get('action_id')!r} does not match "
            f"planned action {action.action_id}."
        )

    achieved = envelope.get("achieved")
    if not isinstance(achieved, Mapping):
        raise StaticCampaignArtifactError("result envelope is missing 'achieved'.")
    if str(achieved.get("stage")) != str(action.stage):
        raise StaticCampaignArtifactError(
            f"achieved stage {achieved.get('stage')!r} != planned {action.stage!r}."
        )
    if str(achieved.get("cell_digest")) != action.cell.digest():
        raise StaticCampaignArtifactError(
            "achieved cell digest does not match the planned cell identity."
        )
    achieved_method = normalize_method_id(str(achieved.get("method_id")))
    if achieved_method != normalize_method_id(action.cell.method_id):
        raise StaticCampaignArtifactError(
            f"achieved method {achieved.get('method_id')!r} normalizes to "
            f"{achieved_method!r} != planned {action.cell.method_id!r}."
        )
    if str(achieved.get("physics_digest")) != action.cell.physics.digest():
        raise StaticCampaignArtifactError(
            "achieved physics digest does not match the planned physics identity."
        )

    declared_files = envelope.get("payload_files")
    declared_digests = envelope.get("output_digests")
    if not isinstance(declared_files, list) or not isinstance(declared_digests, Mapping):
        raise StaticCampaignArtifactError(
            "result envelope must declare payload_files and output_digests."
        )
    verified_digests: dict[str, str] = {}
    for name in declared_files:
        payload_path = out_dir / str(name)
        if not payload_path.is_file():
            raise StaticCampaignArtifactError(
                f"declared payload file {name!r} is missing at {payload_path}."
            )
        actual = sha256_file(payload_path)
        expected = str(declared_digests.get(str(name)) or "")
        if actual != expected:
            raise StaticCampaignArtifactError(
                f"output digest drift for {name!r}: file hash {actual} != "
                f"declared {expected}."
            )
        verified_digests[str(name)] = actual

    qualification = envelope.get("qualification")
    return StaticQSEAttempt.from_mappings(
        action_id=action.action_id,
        achieved_config={
            "cell_digest": str(achieved.get("cell_digest")),
            "stage": str(achieved.get("stage")),
            "method_id": achieved_method,
            "physics_digest": str(achieved.get("physics_digest")),
            "settings_digest": _settings_digest(achieved.get("settings")),
        },
        source_revision=str(envelope.get("source_revision") or "unknown"),
        input_digests={
            str(k): str(v)
            for k, v in (envelope.get("input_digests") or {}).items()
        },
        output_digests=verified_digests,
        payload_paths=[
            str(Path(out_dir.name) / str(name)) for name in declared_files
        ],
        disposition=str(envelope.get("disposition") or "completed"),
        qualification=qualification if isinstance(qualification, Mapping) else None,
    )


def _settings_digest(settings: Any) -> str:
    from pipelines.qse_spectra.static_campaign import canonical_digest

    return canonical_digest(settings if isinstance(settings, Mapping) else {})


def discover_completed_action_ids(artifact_root: Path) -> tuple[str, ...]:
    """Action ids with a result envelope under ``artifact_root``, sorted.

    Discovery applies only to completed *outputs* of declared actions; the
    planner itself never discovers work from the filesystem.
    """

    root = Path(artifact_root)
    if not root.is_dir():
        return ()
    found = [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and (entry / RESULT_ENVELOPE_FILENAME).is_file()
    ]
    return tuple(sorted(found))


@dataclass(frozen=True)
class AdmissionSummary:
    admitted: tuple[str, ...]
    idempotent: tuple[str, ...]
    results: tuple[AdmissionResult, ...]


def record_action_outputs(
    package: Any,
    completed_action_ids: Collection[str],
    ledger: StaticQSELedger,
) -> AdmissionSummary:
    """Verify and admit an exact nonempty subset of a package's actions.

    ``package`` must expose ``actions_by_id`` (mapping action id to
    :class:`StaticQSEAction`) and ``artifact_root``. Unknown ids fail closed;
    verification failures propagate; exact replay is an idempotent no-op;
    conflicting replay raises inside the ledger.
    """

    ids = [str(action_id) for action_id in completed_action_ids]
    if not ids:
        raise StaticCampaignArtifactError(
            "record_action_outputs requires a nonempty action-id subset."
        )
    actions_by_id: Mapping[str, StaticQSEAction] = package.actions_by_id
    unknown = sorted(set(ids) - set(actions_by_id))
    if unknown:
        raise StaticCampaignArtifactError(
            f"unknown action ids for this package: {unknown}."
        )
    admitted: list[str] = []
    idempotent: list[str] = []
    results: list[AdmissionResult] = []
    for action_id in sorted(set(ids)):
        action = actions_by_id[action_id]
        attempt = verify_achieved_action(action, Path(package.artifact_root))
        result = ledger.append(attempt)
        results.append(result)
        if result.admitted:
            admitted.append(action_id)
        elif result.idempotent:
            idempotent.append(action_id)
    return AdmissionSummary(
        admitted=tuple(admitted),
        idempotent=tuple(idempotent),
        results=tuple(results),
    )


__all__ = [
    "AdmissionSummary",
    "RESULT_ENVELOPE_FILENAME",
    "RESULT_ENVELOPE_SCHEMA_VERSION",
    "StaticCampaignArtifactError",
    "action_output_dir",
    "build_result_envelope",
    "discover_completed_action_ids",
    "record_action_outputs",
    "sha256_file",
    "verify_achieved_action",
    "write_result_envelope",
]
