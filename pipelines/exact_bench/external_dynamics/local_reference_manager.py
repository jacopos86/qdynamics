#!/usr/bin/env python3
"""Resolve optional local external-dynamics reference checkouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from pipelines.exact_bench.external_dynamics.provenance import (
    ExternalDynamicsReferenceSpec,
    get_external_dynamics_reference_spec,
)

DEFAULT_REFERENCE_ROOT = Path("tmp/external_parity_refs")
DEFAULT_MANIFEST_PATH = DEFAULT_REFERENCE_ROOT / "manifest.json"
MANIFEST_ENV = "HOLSTEIN_EXTERNAL_DYNAMICS_PARITY_MANIFEST"
ROOT_ENV = "HOLSTEIN_EXTERNAL_DYNAMICS_PARITY_ROOT"


@dataclass(frozen=True)
class ResolvedExternalDynamicsReference:
    reference_id: str
    status: str
    reason: str
    spec: dict[str, Any]
    manifest_path: str | None = None
    checkout_path: str | None = None
    manifest_commit: str | None = None
    resolved_commit: str | None = None
    commit_date: str | None = None
    license_files: tuple[str, ...] = ()
    provenance_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["license_files"] = list(self.license_files)
        return payload


class ExternalDynamicsReferenceError(RuntimeError):
    """Raised for malformed local reference requests."""


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExternalDynamicsReferenceError(f"external dynamics manifest missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExternalDynamicsReferenceError(f"external dynamics manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ExternalDynamicsReferenceError(f"external dynamics manifest must be a JSON object: {path}")
    return payload


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _license_files(path: Path) -> tuple[str, ...]:
    names: list[str] = []
    for child in path.iterdir() if path.exists() else ():
        if child.is_file() and child.name.upper().startswith(("LICENSE", "COPYING")):
            names.append(child.name)
    return tuple(sorted(names))


def default_manifest_path() -> Path:
    raw = os.environ.get(MANIFEST_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_MANIFEST_PATH


def default_reference_root() -> Path:
    raw = os.environ.get(ROOT_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_REFERENCE_ROOT


def _manifest_entry_for_spec(manifest: Mapping[str, Any], spec: ExternalDynamicsReferenceSpec) -> Mapping[str, Any] | None:
    repos = manifest.get("repos", [])
    if not isinstance(repos, list):
        return None
    expected_names = {spec.default_local_subdir, spec.reference_id}
    expected_urls = {spec.url, spec.clone_url}
    for entry in repos:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        url = str(entry.get("url", "")).strip()
        if name in expected_names or url in expected_urls:
            return entry
    return None


def resolve_external_dynamics_reference(
    reference_id: str,
    *,
    manifest_path: str | Path | None = None,
    reference_root: str | Path | None = None,
    allow_unpinned_local_reference: bool = False,
) -> ResolvedExternalDynamicsReference:
    """Resolve one optional local reference checkout into structured provenance.

    Missing or mismatched external references are diagnostic skips, not errors for
    the production benchmark path.
    """

    spec = get_external_dynamics_reference_spec(reference_id)
    if spec.availability == "repo_native_only":
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="not_applicable",
            reason="repo-native diagnostic does not use an external checkout",
            spec=spec.to_dict(),
            provenance_verified=True,
        )

    mpath = Path(manifest_path).expanduser() if manifest_path is not None else default_manifest_path()
    root = Path(reference_root).expanduser() if reference_root is not None else default_reference_root()
    try:
        manifest = _read_json(mpath)
    except ExternalDynamicsReferenceError as exc:
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_missing_manifest",
            reason=str(exc),
            spec=spec.to_dict(),
            manifest_path=str(mpath),
        )

    entry = _manifest_entry_for_spec(manifest, spec)
    if entry is None:
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_missing_manifest_entry",
            reason=f"manifest {mpath} has no repo entry for {spec.reference_id}",
            spec=spec.to_dict(),
            manifest_path=str(mpath),
        )

    raw_path = entry.get("path") or spec.default_local_subdir
    checkout = Path(str(raw_path)).expanduser()
    if not checkout.is_absolute():
        # Prefer manifest path relative to repository root/current working dir;
        # if absent, fall back to the configured reference root.
        candidate = checkout
        if not candidate.exists() and spec.default_local_subdir:
            candidate = root / spec.default_local_subdir
        checkout = candidate
    if not checkout.exists():
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_missing_checkout",
            reason=f"external reference checkout missing: {checkout}",
            spec=spec.to_dict(),
            manifest_path=str(mpath),
            checkout_path=str(checkout),
            manifest_commit=str(entry.get("commit")) if entry.get("commit") else None,
        )

    manifest_commit = str(entry.get("commit", "")).strip() or None
    resolved_commit = _git_commit(checkout)
    licenses = _license_files(checkout)
    if not manifest_commit and not allow_unpinned_local_reference:
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_unpinned_reference",
            reason=f"manifest entry for {spec.reference_id} has no commit pin",
            spec=spec.to_dict(),
            manifest_path=str(mpath),
            checkout_path=str(checkout),
            manifest_commit=manifest_commit,
            resolved_commit=resolved_commit,
            commit_date=str(entry.get("commit_date")) if entry.get("commit_date") else None,
            license_files=licenses,
        )
    if manifest_commit and not resolved_commit and not allow_unpinned_local_reference:
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_provenance_unresolved",
            reason=f"could not resolve git commit for {checkout}",
            spec=spec.to_dict(),
            manifest_path=str(mpath),
            checkout_path=str(checkout),
            manifest_commit=manifest_commit,
            resolved_commit=resolved_commit,
            commit_date=str(entry.get("commit_date")) if entry.get("commit_date") else None,
            license_files=licenses,
        )
    if manifest_commit and resolved_commit and manifest_commit != resolved_commit:
        return ResolvedExternalDynamicsReference(
            reference_id=spec.reference_id,
            status="skipped_provenance_mismatch",
            reason=f"resolved commit {resolved_commit} does not match manifest commit {manifest_commit}",
            spec=spec.to_dict(),
            manifest_path=str(mpath),
            checkout_path=str(checkout),
            manifest_commit=manifest_commit,
            resolved_commit=resolved_commit,
            commit_date=str(entry.get("commit_date")) if entry.get("commit_date") else None,
            license_files=licenses,
        )

    return ResolvedExternalDynamicsReference(
        reference_id=spec.reference_id,
        status="available",
        reason="external reference checkout resolved",
        spec=spec.to_dict(),
        manifest_path=str(mpath),
        checkout_path=str(checkout),
        manifest_commit=manifest_commit,
        resolved_commit=resolved_commit,
        commit_date=str(entry.get("commit_date")) if entry.get("commit_date") else None,
        license_files=licenses,
        provenance_verified=bool(manifest_commit and resolved_commit == manifest_commit),
    )
