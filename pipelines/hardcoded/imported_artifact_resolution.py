#!/usr/bin/env python3
"""Shared helpers for resolving imported HH/ADAPT artifact sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ImportedArtifactResolution:
    mode: str
    requested_json: Path | None
    resolved_json: Path | None
    source_kind: str | None
    default_subject: bool = False


def _load_payload(json_path: Path) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Imported artifact payload at {json_path} is not a JSON object.")
    return dict(payload)


def resolve_default_lean_pareto_l2_artifact_json() -> tuple[Path | None, bool]:
    repo_root = Path(__file__).resolve().parents[2]
    currentcode = sorted(
        repo_root.glob(
            "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_currentcode_*.json"
        )
    )
    if currentcode:
        return Path(currentcode[-1]), True
    legacy = repo_root / "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell.json"
    if legacy.exists():
        return legacy, True
    return None, False


def resolve_imported_artifact_path(
    *,
    requested_json: Path | None,
    require_default_import_source: bool,
) -> ImportedArtifactResolution:
    requested = Path(requested_json) if requested_json is not None else None
    default_subject = False
    if requested is None and bool(require_default_import_source):
        requested, default_subject = resolve_default_lean_pareto_l2_artifact_json()
    if requested is None:
        return ImportedArtifactResolution(
            mode="fresh_stage",
            requested_json=None,
            resolved_json=None,
            source_kind=None,
            default_subject=False,
        )

    requested = Path(requested)
    if not requested.exists():
        raise FileNotFoundError(f"Imported noise-audit source not found: {requested}")

    payload = _load_payload(requested)
    resolved = requested
    source_kind = "direct_payload"
    if not isinstance(payload.get("adapt_vqe", None), Mapping):
        artifacts = payload.get("artifacts", {}) if isinstance(payload, Mapping) else {}
        intermediate = artifacts.get("intermediate", {}) if isinstance(artifacts, Mapping) else {}
        sidecar = intermediate.get("adapt_handoff_json", None) if isinstance(intermediate, Mapping) else None
        if sidecar is not None:
            resolved = Path(sidecar)
            if not resolved.is_absolute():
                resolved = (requested.parent / resolved).resolve()
            source_kind = "workflow_sidecar"

    return ImportedArtifactResolution(
        mode="imported_artifact",
        requested_json=requested,
        resolved_json=resolved,
        source_kind=source_kind,
        default_subject=bool(default_subject),
    )
