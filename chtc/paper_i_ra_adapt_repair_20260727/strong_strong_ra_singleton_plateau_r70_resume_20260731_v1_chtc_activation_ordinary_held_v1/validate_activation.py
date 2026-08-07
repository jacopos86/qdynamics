#!/usr/bin/env python3
"""Validate the exact one-row strong-strong RA-plateau activation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_EXECUTION_ID = "core__strong_strong_u8__nph7__ra_singleton_plateau__r70"
EXPECTED_SCIENTIFIC_SHA = "ff2737c13ce54b0b3ff9dda9ad7a747623c14c093fb4fe3ce9d27878c570c916"
EXPECTED_AUTH_CANONICAL_SHA = "ee8576fac8d7514b883c74c4e9dc19c23dfccaf89ccc1406f757329cde5edb05"
EXPECTED_JOB_CANONICAL_SHA = "3a0b316158e9ee5e3a424e4bf5558845b7d28e90bf00113c2125a4f94ce0081c"


class ActivationError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ActivationError(f"JSON object required: {path}")
    return value


def verify_self(payload: dict[str, Any], *, label: str) -> str:
    claimed = payload.get("sha256")
    body = dict(payload)
    body.pop("sha256", None)
    observed = hashlib.sha256(canonical_bytes(body)).hexdigest()
    if claimed != observed:
        raise ActivationError(f"{label} canonical SHA-256 mismatch")
    return observed


def validate(repo_root: Path, *, require_image: bool) -> dict[str, Any]:
    activation = Path(__file__).resolve().parent
    manifest = load_json(activation / "activation_manifest.json")
    verify_self(manifest, label="activation manifest")
    if (
        manifest.get("schema") != "paper_i_ra_adapt_ss_singleton_plateau_r70_activation_manifest_v1"
        or manifest.get("status") != "passed_one_row_ordinary_held_activation"
        or manifest.get("execution_id") != EXPECTED_EXECUTION_ID
        or manifest.get("row_count") != 1
        or manifest.get("initially_held") is not True
        or manifest.get("automatic_release") is not False
        or manifest.get("strong_weak_rows_included") is not False
        or manifest.get("scientific_settings_sha256") != EXPECTED_SCIENTIFIC_SHA
    ):
        raise ActivationError("activation contract drifted")
    for binding in manifest.get("bindings", []):
        path = repo_root / str(binding.get("path", ""))
        if binding.get("role") == "image" and not path.exists() and not require_image:
            continue
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(binding.get("size_bytes", -1))
            or sha256_file(path) != binding.get("sha256")
        ):
            raise ActivationError(f"binding drifted: {binding.get('path')}")
    auth = load_json(activation / "execution_authorization.json")
    job = load_json(
        repo_root
        / "chtc/paper_i_ra_adapt_repair_20260727/strong_strong_ra_singleton_plateau_r70_fastpath_v1_runtime/jobs_v2"
        / f"{EXPECTED_EXECUTION_ID}.json"
    )
    if (
        verify_self(auth, label="authorization") != EXPECTED_AUTH_CANONICAL_SHA
        or verify_self(job, label="job") != EXPECTED_JOB_CANONICAL_SHA
        or auth.get("execution_id") != EXPECTED_EXECUTION_ID
        or auth.get("execution_authorized") is not True
        or auth.get("submission_authorized") is not False
        or job.get("scientific_settings_sha256") != EXPECTED_SCIENTIFIC_SHA
        or job.get("execution_mode") != "authenticated_resume_50_to_70"
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("active_gradient_policy") != "stationary_source_response_v1"
        or job.get("resource_weighting_scope") != "late_resource_weighting_v1"
        or int(job.get("source_horizon", -1)) != 50
        or int(job.get("target_horizon", -1)) != 70
    ):
        raise ActivationError("authorization or scientific row drifted")
    return {"status": "passed", "execution_id": EXPECTED_EXECUTION_ID, "manifest_sha256": manifest["sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--require-image", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.repo_root.resolve(), require_image=args.require_image), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
