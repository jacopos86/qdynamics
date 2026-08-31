#!/usr/bin/env python3
"""Safe fetch/pin helper for external benchmark reference repositories.

Only cataloged repositories may be cloned.  Checkouts live outside this repo by
default so generated third-party code is not staged or imported accidentally.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pipelines.exact_bench.external_adapt.provenance import (
    get_external_reference_spec,
    reference_catalog,
)

DEFAULT_CACHE_ENV = "HOLSTEIN_EXTERNAL_COMPETITOR_CACHE"
DEFAULT_CACHE_ROOT = Path("~/.cache/holstein_external_competitors")
LOCKFILE_NAME = "external_adapt_reference_lock.json"


@dataclass(frozen=True)
class MaterializedReference:
    reference_id: str
    display_name: str
    url: str
    clone_url: str | None
    checkout_dir: str | None
    requested_ref: str | None
    resolved_commit: str | None
    materialized_utc: str
    status: str
    reason: str = ""
    license_files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_cache_root() -> Path:
    """Return the cache root for external checkouts."""

    raw = os.environ.get(DEFAULT_CACHE_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_CACHE_ROOT.expanduser()


def checkout_dir_for(reference_id: str, *, cache_root: Path | str | None = None) -> Path:
    spec = get_external_reference_spec(reference_id)
    root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    return root / spec.reference_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(args: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *map(str, args)],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _license_files(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    names: list[str] = []
    for child in path.iterdir():
        if child.is_file() and child.name.lower().startswith(("license", "copying")):
            names.append(child.name)
    return tuple(sorted(names))


def materialize_reference(
    reference_id: str,
    *,
    cache_root: Path | str | None = None,
    ref: str | None = None,
    fetch: bool = True,
) -> MaterializedReference:
    """Clone/fetch a cataloged reference and return resolved provenance.

    This function never accepts arbitrary URLs.  It uses only the URL recorded in
    ``provenance.reference_catalog()``.
    """

    spec = get_external_reference_spec(reference_id)
    requested_ref = str(ref or spec.default_ref or "HEAD")
    if spec.clone_url is None:
        return MaterializedReference(
            reference_id=spec.reference_id,
            display_name=spec.display_name,
            url=spec.url,
            clone_url=None,
            checkout_dir=None,
            requested_ref=requested_ref,
            resolved_commit=None,
            materialized_utc=_utc_now(),
            status="skipped_no_clone_url",
            reason="catalog entry is not a public git checkout",
        )
    target = checkout_dir_for(spec.reference_id, cache_root=cache_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        _run_git(["clone", "--no-tags", spec.clone_url, str(target)])
    elif fetch:
        _run_git(["fetch", "--prune", "origin"], cwd=target)
    # Prefer an exact ref if provided; otherwise use the catalog default.  The
    # resolved commit is what downstream manifests should cite.
    _run_git(["checkout", requested_ref], cwd=target)
    if fetch and requested_ref in {"main", "master"}:
        # Pull the named default branch only after checkout.  This keeps branch
        # defaults explicit while still allowing current lock generation.
        try:
            _run_git(["pull", "--ff-only", "origin", requested_ref], cwd=target)
        except subprocess.CalledProcessError:
            # Some repos use detached/default refs; the commit hash below remains authoritative.
            pass
    resolved = _run_git(["rev-parse", "HEAD"], cwd=target)
    return MaterializedReference(
        reference_id=spec.reference_id,
        display_name=spec.display_name,
        url=spec.url,
        clone_url=spec.clone_url,
        checkout_dir=str(target),
        requested_ref=requested_ref,
        resolved_commit=resolved,
        materialized_utc=_utc_now(),
        status="materialized",
        license_files=_license_files(target),
    )


def write_reference_lock(
    materialized: Sequence[MaterializedReference],
    *,
    cache_root: Path | str | None = None,
    lockfile_name: str = LOCKFILE_NAME,
) -> Path:
    """Write a generated lockfile beside the external checkout cache."""

    root = Path(cache_root).expanduser() if cache_root is not None else default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / str(lockfile_name)
    payload = {
        "schema": "external_adapt_reference_lock_v1",
        "generated_utc": _utc_now(),
        "cache_root": str(root),
        "references": [item.to_dict() for item in materialized],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def catalog_payload() -> dict[str, Any]:
    """Return catalog metadata as JSON-serializable data."""

    return {
        "schema": "external_adapt_reference_catalog_v1",
        "cache_env": DEFAULT_CACHE_ENV,
        "default_cache_root": str(default_cache_root()),
        "references": [spec.to_dict() for spec in reference_catalog()],
    }


__all__ = [
    "DEFAULT_CACHE_ENV",
    "LOCKFILE_NAME",
    "MaterializedReference",
    "catalog_payload",
    "checkout_dir_for",
    "default_cache_root",
    "materialize_reference",
    "write_reference_lock",
]
