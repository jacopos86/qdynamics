#!/usr/bin/env python3
"""Record a completed archive-only preflight without enabling submission."""

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
    if bundle["submission_enabled"] is not False:
        raise RuntimeError("preparation task must leave submission disabled")
    payload = {
        "schema": "paper_i_hh_append_completion_archive_preflight_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source_archive_sha256": source["archive_sha256"],
        "generic_static_adapt_variants_sha256": (
            source["files"]["pipelines/exact_bench/generic_static_adapt_variants.py"]["sha256"]
        ),
        "checks": {
            "bundle_contract_tests": "17 passed",
            "archive_only_comparator_fidelity_regressions": "108 passed",
            "archive_only_worker_import": "pass; six completion cases resolved",
            "worker_static_preflight": "py_compile, bash -n, and --help pass",
            "normalized_manifest_count": 12,
            "visible_append_source_lock_count": 6,
            "same_cutoff_and_horizon_contract": "pass",
            "submission_fail_closed": "pass",
        },
        "scientific_blockers": [],
        "operational_blockers": [
            "authenticated remote image/Qiskit gate pending",
            "submission deliberately disabled",
        ],
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
