#!/usr/bin/env python3
"""Paper III production-contract payload builders for the QSE CLI.

RECONSTRUCTION (2026-08-17): the original module was never committed and was
lost when snapshot commit 6442fbb5 captured ``__main__.py`` without its
siblings. This reconstruction implements the exact surface witnessed by
``__main__.py`` (imports, call signatures, and CLI gating) per
``prompt-exports/paper_iii_qse_qiskit_cost_integration_and_route_repair_spec_20260817.md``
in the Documents clone. ``run_class`` values: "diagnostic" is observed in
stored 20260802 manifests, "candidate" is the CLI default; "paper_final" is
the reconstructed production tier.

The contract is emitted only under ``--paper-iii-static-qse-mode`` and never
alters spectra; it records run-class, visible claim target, compatibility
tier, and (for HH full_meta pools) file-hashed basis/seed provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

PAPER_III_PRODUCTION_CONTRACT_SCHEMA_VERSION = "paper_iii_production_contract_v1"
HH_FULL_META_OPERATOR_BASIS_SOURCES = frozenset({"full_meta", "full_meta_filtered"})
PAPER_III_RUN_CLASSES = ("diagnostic", "candidate", "paper_final")


class PaperIIIProductionContractError(ValueError):
    """Raised when Paper III production-contract inputs are missing or contradictory."""


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_existing_file(path: Any, *, flag: str) -> Path:
    if path is None:
        raise PaperIIIProductionContractError(
            f"Paper III production contract requires {flag}; none was supplied."
        )
    resolved = Path(path)
    if not resolved.is_file():
        raise PaperIIIProductionContractError(
            f"Paper III production contract {flag} does not exist: {resolved}"
        )
    return resolved


def _seed_manifest_pool_key(payload: Mapping[str, Any]) -> str | None:
    for key in ("pool_key", "operator_basis_source"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    operator_basis = payload.get("operator_basis")
    if isinstance(operator_basis, Mapping):
        value = operator_basis.get("pool_key")
        if isinstance(value, str) and value:
            return value
    return None


def resolve_hh_full_meta_provenance(
    *,
    operator_basis_source: str,
    basis_artifact_path: Any,
    seed_manifest_path: Any,
    cli_pool_key: str,
    basis_provenance: Mapping[str, Any] | None,
    hamiltonian_source: Mapping[str, Any] | None,
    state_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve and hash the HH full_meta basis/seed provenance for the contract."""

    source = str(operator_basis_source)
    if source not in HH_FULL_META_OPERATOR_BASIS_SOURCES:
        raise PaperIIIProductionContractError(
            "HH full_meta provenance requires operator_basis_source in "
            f"{sorted(HH_FULL_META_OPERATOR_BASIS_SOURCES)!r}; got {source!r}."
        )
    basis_file = _require_existing_file(basis_artifact_path, flag="--basis-artifact-json")
    seed_file = _require_existing_file(seed_manifest_path, flag="--seed-manifest-json")
    try:
        seed_payload = json.loads(seed_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperIIIProductionContractError(
            f"Paper III seed manifest is not readable JSON: {seed_file} ({exc})"
        ) from exc
    if not isinstance(seed_payload, Mapping):
        raise PaperIIIProductionContractError(
            f"Paper III seed manifest must be a JSON object: {seed_file}"
        )
    declared_pool_key = _seed_manifest_pool_key(seed_payload)
    if declared_pool_key is not None and str(cli_pool_key) != declared_pool_key:
        raise PaperIIIProductionContractError(
            "Paper III seed manifest pool key contradicts the CLI pool key: "
            f"manifest declares {declared_pool_key!r}, CLI requested {cli_pool_key!r}."
        )
    return {
        "operator_basis_source": source,
        "cli_pool_key": str(cli_pool_key),
        "basis_artifact_path": str(basis_file),
        "basis_artifact_sha256": _sha256_of_file(basis_file),
        "seed_manifest_path": str(seed_file),
        "seed_manifest_sha256": _sha256_of_file(seed_file),
        "seed_manifest_pool_key": declared_pool_key,
        "basis_provenance": dict(basis_provenance) if basis_provenance is not None else None,
        "hamiltonian_source": dict(hamiltonian_source) if hamiltonian_source is not None else None,
        "state_source": dict(state_source) if state_source is not None else None,
    }


def build_paper_iii_contract(
    *,
    run_class: str,
    visible_target: str,
    compatibility_tier: str,
    hh_full_meta_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the Paper III production-contract manifest payload."""

    resolved_run_class = str(run_class)
    if resolved_run_class not in PAPER_III_RUN_CLASSES:
        raise PaperIIIProductionContractError(
            f"Paper III run class must be one of {list(PAPER_III_RUN_CLASSES)!r}; "
            f"got {resolved_run_class!r}."
        )
    resolved_target = str(visible_target).strip()
    if not resolved_target:
        raise PaperIIIProductionContractError("Paper III visible target must be a non-empty string.")
    resolved_tier = str(compatibility_tier).strip()
    if not resolved_tier:
        raise PaperIIIProductionContractError(
            "Paper III compatibility tier must be a non-empty string."
        )
    return {
        "schema_version": PAPER_III_PRODUCTION_CONTRACT_SCHEMA_VERSION,
        "reconstructed_module": True,
        "run_class": resolved_run_class,
        "visible_target": resolved_target,
        "compatibility_tier": resolved_tier,
        "hh_full_meta_provenance": (
            dict(hh_full_meta_provenance) if hh_full_meta_provenance is not None else None
        ),
    }
