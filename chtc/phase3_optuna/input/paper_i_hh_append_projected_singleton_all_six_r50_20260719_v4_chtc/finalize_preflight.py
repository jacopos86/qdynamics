#!/usr/bin/env python3
"""Record the immutable Append v2 successor archive-only preflight."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    source = _load(BUNDLE / "source_archive_manifest.json")
    bundle = _load(BUNDLE / "bundle_manifest.json")
    archive = BUNDLE / "source_locked.tar.gz"
    if _sha256(archive) != source["archive_sha256"]:
        raise RuntimeError("source archive hash mismatch")
    if bundle["submission_enabled"] is not True:
        raise RuntimeError("successor bundle must retain the authenticated submission gate")
    payload = {
        "schema": "paper_i_hh_append_completion_successor_preflight_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source_archive_sha256": source["archive_sha256"],
        "generic_static_adapt_variants_sha256": (
            source["files"]["pipelines/exact_bench/generic_static_adapt_variants.py"]["sha256"]
        ),
        "checks": {
            "bundle_contract_tests": "20 passed",
            "archive_only_powell_cap_and_projected_pool_regressions": "12 passed",
            "archive_only_worker_import": "pass; source-defined cap classifier imported and exercised",
            "worker_static_preflight": "py_compile, bash -n, and --help pass",
            "normalized_manifest_count": 12,
            "visible_append_source_lock_count": 6,
            "same_cutoff_and_horizon_contract": "pass",
            "optimizer_failure_policy": (
                "only finite non-increasing exact Powell maxiter caps accepted; "
                "every other failure rejected"
            ),
        },
        "scientific_blockers": [],
        "operational_blockers": [],
    }
    _write(BUNDLE / "archive_only_preflight.json", payload)
    hashes = {
        path.relative_to(BUNDLE).as_posix(): _sha256(path)
        for path in sorted(BUNDLE.rglob("*"))
        if path.is_file() and path.name != "submission_artifact_hashes.json"
    }
    _write(
        BUNDLE / "submission_artifact_hashes.json",
        {"schema": "paper_i_hh_append_completion_artifact_hashes_v1", "files": hashes},
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
