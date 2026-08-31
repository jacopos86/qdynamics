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


def resolve_default_lean_pareto_l2_circuit_ready_artifact_json() -> tuple[Path | None, bool]:
    repo_root = Path(__file__).resolve().parents[2]
    circuit_ready = sorted(
        repo_root.glob(
            "artifacts/json/adapt_hh_L2_ecut1_pareto_lean_l2_phase3_powell_rerun_with_ansatz_input_*.json"
        )
    )
    if circuit_ready:
        return Path(circuit_ready[-1]), True
    return None, False


def resolve_default_hh_nighthawk_fixed_scaffold_artifact_json() -> tuple[Path | None, bool]:
    """Resolve the local fake-backend fixed-scaffold default HH artifact.

    This stays pinned to the original Nighthawk gate-pruned 7-term line used by the
    local compile-control and attribution scout slices.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gate_pruned = repo_root / "artifacts/json/hh_prune_nighthawk_gate_pruned_7term.json"
    if gate_pruned.exists():
        return gate_pruned, True
    circuit_opt = repo_root / "artifacts/json/hh_prune_nighthawk_circuit_optimized_7term.json"
    if circuit_opt.exists():
        return circuit_opt, True
    return None, False


def resolve_default_hh_marrakesh_runtime_candidate_artifact_json() -> tuple[Path | None, bool]:
    """Resolve the honest default imported runtime candidate HH artifact.

    Prefers the exported Marrakesh/Heron runtime candidate: the gate-pruned 6-term
    drop-eyezee scaffold selected for real Runtime/QPU replay work.
    """
    repo_root = Path(__file__).resolve().parents[2]
    runtime_candidate = sorted(
        repo_root.glob(
            "artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_*.json"
        )
    )
    if runtime_candidate:
        return Path(runtime_candidate[-1]), True
    fallback = repo_root / "artifacts/json/hh_marrakesh_fixed_scaffold_6term_drop_eyezee_20260323T171528Z.json"
    if fallback.exists():
        return fallback, True
    return None, False


def resolve_default_hh_nighthawk_circuit_optimized_7term_artifact_json() -> tuple[Path | None, bool]:
    """Backward-compatible alias for the local fixed-scaffold resolver."""
    return resolve_default_hh_nighthawk_fixed_scaffold_artifact_json()


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
