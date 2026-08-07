from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = REPAIR_ROOT / "stationary_ra_always12_r50_20260729_v1_chtc"
QUARANTINE_DIR = REPAIR_ROOT / "quarantine"
TOMBSTONE_PATH = (
    QUARANTINE_DIR
    / "stationary_ra_always12_r50_20260729_v1_quarantine.json"
)
QUARANTINED_AUTHORIZATION_PATH = (
    QUARANTINE_DIR
    / "stationary_ra_always12_r50_20260729_v1_"
    "submission_authorization_receipt.invalid.json"
)

EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "591e91cc220915acad6382e7ffc29c10e745522870c2e767a8048d15f1c67cd6"
)
EXPECTED_PACKAGE_MANIFEST_FILE_SHA256 = (
    "73e1a41cf8e0920b1970957e6ab2415a5f7d3771429eb22d29cf393b4a835ae7"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "387fde6ce8cb241860a0b942fbd43cfefcb26f9dc501ac7d8896d1beb98d1a0f"
)
EXPECTED_QUARANTINED_AUTHORIZATION_FILE_SHA256 = (
    "70b7e013a902bc7db9f792de0e08d0b2fc44254029985d2426ac448d34d1e66b"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalid_v1_is_quarantined_without_mutating_sealed_payloads() -> None:
    tombstone = _load(TOMBSTONE_PATH)

    assert tombstone["schema"] == (
        "paper_i_ra_adapt_invalid_package_quarantine_v1"
    )
    assert tombstone["invalid_reason"] == (
        "raw_unreduced_adapt_insertion_mode_full"
    )
    assert tombstone["package_manifest_sha256"] == (
        EXPECTED_PACKAGE_MANIFEST_SHA256
    )
    assert tombstone["package_manifest_file_sha256"] == (
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256
    )
    assert tombstone["source_archive_sha256"] == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )
    assert tombstone["paper_i_evidence_eligible"] is False
    assert tombstone["execution_authorized"] is False
    assert tombstone["submission_authorized"] is False
    assert tombstone["submitted"] is False
    assert tombstone["required_action"] == "do_not_execute_or_submit"
    assert tombstone["scheduler_contacted"] is False

    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    assert manifest["sha256"] == EXPECTED_PACKAGE_MANIFEST_SHA256
    assert _file_sha256(PACKAGE_DIR / "package_manifest.json") == (
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256
    )
    assert manifest["source_archive"]["sha256"] == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )
    assert _file_sha256(PACKAGE_DIR / "source_locked.tar.gz") == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )


def test_invalid_v1_authorization_overlay_is_preserved_but_unreachable() -> None:
    original_path = (
        PACKAGE_DIR / "authority/submission_authorization_receipt.json"
    )
    assert not original_path.exists()
    assert _file_sha256(QUARANTINED_AUTHORIZATION_PATH) == (
        EXPECTED_QUARANTINED_AUTHORIZATION_FILE_SHA256
    )

    tombstone = _load(TOMBSTONE_PATH)
    removed = tombstone["removed_mutable_authority_overlay"]
    assert removed["original_path"] == (
        "authority/submission_authorization_receipt.json"
    )
    assert removed["file_sha256"] == (
        EXPECTED_QUARANTINED_AUTHORIZATION_FILE_SHA256
    )
    assert removed["exact_bytes_preserved"] is True

    quarantined = _load(QUARANTINED_AUTHORIZATION_PATH)
    assert quarantined["execution_authorized"] is True
    assert quarantined["submission_authorized"] is True
    assert quarantined["submission_state"] == "authorized_not_submitted"

    wrapper = (PACKAGE_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    submit = (PACKAGE_DIR / "submit.sub").read_text(encoding="utf-8")
    required_overlay = (
        "${package_dir}/authority/submission_authorization_receipt.json"
    )
    assert required_overlay in wrapper
    assert (
        "Required package member: "
        "authority/submission_authorization_receipt.json"
    ) in submit


def test_invalid_v1_records_the_retired_raw_insertion_contract() -> None:
    plan = _load(PACKAGE_DIR / "execution_plan.json")
    observed_kinds: set[str] = set()
    observed_modes: set[str] = set()

    for execution in plan["direct_executions"]:
        job = _load(PACKAGE_DIR / execution["job_spec_path"])
        protocol = _load(REPO_ROOT / job["protocol"]["path"])
        observed_kinds.add(protocol["request"]["method"]["insertion"]["kind"])
        observed_modes.add(
            protocol["route_contract"]["execution_settings"][
                "adapt_insertion_mode"
            ]
        )

    assert observed_kinds == {"full_commutation"}
    assert observed_modes == {"full"}
    assert _load(TOMBSTONE_PATH)["superseding_package_id"] == (
        "paper_i_ra_adapt_stationary_ra_always12_r50_20260729_v2_chtc"
    )
