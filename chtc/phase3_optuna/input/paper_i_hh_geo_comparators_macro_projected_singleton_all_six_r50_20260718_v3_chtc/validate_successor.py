#!/usr/bin/env python3
"""Fail-closed, non-scientific validation for the Geo v3 successor bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
EXPECTED_BUNDLE_ID = (
    "paper_i_hh_geo_comparators_macro_projected_singleton_all_six_"
    "r50_20260718_v3_chtc"
)
EXPECTED_SOURCE_SHA256 = "8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_id") != EXPECTED_BUNDLE_ID:
        raise RuntimeError("Geo successor bundle identity mismatch")
    if manifest.get("status") != "prepared_not_submitted":
        raise RuntimeError("Geo successor preparation status mismatch")
    if _sha256(BUNDLE / "source_locked.tar.gz") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Geo successor source archive hash mismatch")
    worker = (BUNDLE / "run_job.py").read_text(encoding="utf-8")
    for forbidden in (
        "initial_selected_operator_labels=",
        "initial_selected_operator_batches=",
        "initial_theta=",
        "initial_adapt_history=",
    ):
        if forbidden in worker:
            raise RuntimeError(f"Geo successor retained continuation kwarg: {forbidden}")
    wrapper = (BUNDLE / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    if 'python "chtc/phase3_optuna/input/${bundle_id}/run_job.py"' not in wrapper:
        raise RuntimeError("Geo successor worker path is not bundle-local")
    hashes = json.loads((BUNDLE / "submission_artifact_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in hashes["files"].items():
        if _sha256(BUNDLE / relative) != expected:
            raise RuntimeError(f"Geo successor artifact hash mismatch: {relative}")
    print(json.dumps({"status": "pass", "bundle_id": EXPECTED_BUNDLE_ID}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
