#!/usr/bin/env python3
"""Verify the immutable v6 source-only repair; never rebuild or submit it."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = (
    "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_"
    "noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v6_chtc"
)
PREDECESSOR_BUNDLE_ID = BUNDLE_ID.replace("_v6_chtc", "_v5_chtc")
PREDECESSOR_BUNDLE_DIR = BUNDLE_DIR.parent / PREDECESSOR_BUNDLE_ID
PROFILE_CONTRACT_SHA256 = (
    "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050"
)
PREDECESSOR_SOURCE_ARCHIVE_SHA256 = (
    "d7ce13820d6b59bbe01a4dade40b304daa4e9a0c8e705f5ed19457561c524bd1"
)
SOURCE_ARCHIVE_SHA256 = (
    "f11607321e426d73627910a1da76a22a96f4d4bd82f66708b5b202b2e5a61453"
)
EMPTY_GRAM_OVERLAY_SHA256 = (
    "fcbba1e050c894316eeba7bf4468aabbd7e64be69a6d22ab7ebe72add05cc478"
)
REPAIRED_SOURCE_SHA256 = {
    "pipelines/static_adapt/selector_query_closure.py": (
        "bcdd57c621f0b9688ae97537e63dcfa804371115e871a2b57543668fabd73b34"
    ),
    "pipelines/static_adapt/adapt_pipeline.py": (
        "290507efc3690dd8ea0cb204ba26f7b00480f315f9d14eb7194094aed18a1b4e"
    ),
}
WORKSPACE_OVERLAY_SHA256 = (
    "1da782d0d14355f22de3152961ce7f1d12f602920464fec916362f4643785bdf"
)
WORKSPACE_REPAIRED_SOURCE_SHA256 = {
    "pipelines/scaffold/hh_continuation_scoring.py": (
        "3684ae46dfd07cc5f0fdfc8809e29dc47bf1709da62754c88acf5cc3500b5bf8"
    ),
}
SOURCE_LOCK_STATE = (
    "frozen_phase3_batch3_hysteresis_disabled_v4_plus_serialized_"
    "zero_extent_matrix_receipt_repair_v5_plus_batch_selector_workspace_"
    "receipt_repair_v6"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> bool:
    archive = BUNDLE_DIR / "source_locked.tar.gz"
    assert _sha(archive) == SOURCE_ARCHIVE_SHA256
    assert (
        _sha(PREDECESSOR_BUNDLE_DIR / "source_locked.tar.gz")
        == PREDECESSOR_SOURCE_ARCHIVE_SHA256
    )
    assert _sha(BUNDLE_DIR / "empty_gram_receipt_overlay.patch") == (
        EMPTY_GRAM_OVERLAY_SHA256
    )
    assert _sha(BUNDLE_DIR / "batch_selector_workspace_receipt_overlay.patch") == (
        WORKSPACE_OVERLAY_SHA256
    )

    archive_manifest_path = BUNDLE_DIR / "source_archive_manifest.json"
    revision_manifest_path = BUNDLE_DIR / "source_revision_manifest.json"
    archive_manifest = json.loads(archive_manifest_path.read_text())
    revision_manifest = json.loads(revision_manifest_path.read_text())
    assert archive_manifest["archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert archive_manifest["file_count"] == 386
    repair = archive_manifest["empty_gram_receipt_repair"]
    assert repair == revision_manifest["empty_gram_receipt_repair"]
    assert repair["scientific_settings_changed"] is False
    assert repair["route_contract_sha256"] == PROFILE_CONTRACT_SHA256
    assert repair["overlay_sha256"] == EMPTY_GRAM_OVERLAY_SHA256
    assert repair["derived_file_sha256"] == REPAIRED_SOURCE_SHA256
    assert repair["successor_source_archive_sha256"] == (
        PREDECESSOR_SOURCE_ARCHIVE_SHA256
    )
    workspace_repair = archive_manifest[
        "batch_selector_workspace_receipt_repair"
    ]
    assert workspace_repair == revision_manifest[
        "batch_selector_workspace_receipt_repair"
    ]
    assert workspace_repair["scientific_settings_changed"] is False
    assert workspace_repair["selected_records_changed"] is False
    assert workspace_repair["model_inputs_or_selection_changed"] is False
    assert workspace_repair["route_contract_sha256"] == PROFILE_CONTRACT_SHA256
    assert workspace_repair["overlay_sha256"] == WORKSPACE_OVERLAY_SHA256
    assert workspace_repair["derived_file_sha256"] == (
        WORKSPACE_REPAIRED_SOURCE_SHA256
    )
    assert workspace_repair["predecessor_source_archive_sha256"] == (
        PREDECESSOR_SOURCE_ARCHIVE_SHA256
    )
    assert workspace_repair["successor_source_archive_sha256"] == (
        SOURCE_ARCHIVE_SHA256
    )
    assert archive_manifest["worker_source_mode"] == SOURCE_LOCK_STATE

    with tarfile.open(archive, "r:gz") as handle:
        file_members = sorted(
            (member for member in handle if member.isfile()),
            key=lambda member: member.name,
        )
        assert len(file_members) == 386
        for member in file_members:
            data = handle.extractfile(member).read()
            record = archive_manifest["files"][member.name]
            assert hashlib.sha256(data).hexdigest() == record["sha256"]
            assert len(data) == record["size_bytes"]
        selector = handle.extractfile(
            "pipelines/static_adapt/selector_query_closure.py"
        ).read()
        adapt = handle.extractfile(
            "pipelines/static_adapt/adapt_pipeline.py"
        ).read()
        scoring = handle.extractfile(
            "pipelines/scaffold/hh_continuation_scoring.py"
        ).read()
    assert hashlib.sha256(selector).hexdigest() == REPAIRED_SOURCE_SHA256[
        "pipelines/static_adapt/selector_query_closure.py"
    ]
    assert hashlib.sha256(adapt).hexdigest() == REPAIRED_SOURCE_SHA256[
        "pipelines/static_adapt/adapt_pipeline.py"
    ]
    assert hashlib.sha256(scoring).hexdigest() == (
        WORKSPACE_REPAIRED_SOURCE_SHA256[
            "pipelines/scaffold/hh_continuation_scoring.py"
        ]
    )

    with tarfile.open(
        PREDECESSOR_BUNDLE_DIR / "source_locked.tar.gz", "r:gz"
    ) as predecessor_handle:
        predecessor_files = {
            member.name: predecessor_handle.extractfile(member).read()
            for member in predecessor_handle.getmembers()
            if member.isfile()
        }
    with tarfile.open(archive, "r:gz") as successor_handle:
        successor_files = {
            member.name: successor_handle.extractfile(member).read()
            for member in successor_handle.getmembers()
            if member.isfile()
        }
    assert predecessor_files.keys() == successor_files.keys()
    changed_files = {
        relative
        for relative in predecessor_files
        if predecessor_files[relative] != successor_files[relative]
    }
    assert changed_files == {"pipelines/scaffold/hh_continuation_scoring.py"}

    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == BUNDLE_ID
        assert (
            job["route_identity"]["profile_contract_sha256"]
            == PROFILE_CONTRACT_SHA256
        )
        assert job["source_lock"]["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
        assert job["source_lock"]["empty_gram_receipt_repair"] == repair
        assert job["source_lock"][
            "batch_selector_workspace_receipt_repair"
        ] == workspace_repair
        assert job["source_lock"]["worker_source_mode"] == SOURCE_LOCK_STATE
        assert job["source_lock"]["source_archive_manifest_sha256"] == _sha(
            archive_manifest_path
        )
        assert job["source_lock"]["source_revision_manifest_sha256"] == _sha(
            revision_manifest_path
        )
        settings = job["route_identity"]["profile_contract"]["execution_settings"]
        assert settings["phase_live_hysteresis_enabled"] is False
        assert "--phase-live-hysteresis-disabled" in job["command"]["argv"]
        assert int(job["segment"]["target_controller_round"]) == 50
    return True


if __name__ == "__main__":
    verify()
    print("immutable v6 source-only repair verification passed")
