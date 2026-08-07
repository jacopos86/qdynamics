#!/usr/bin/env python3
"""Verify this immutable successor; it intentionally does not rebuild it."""
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_appendix_fs_prune_nodamping_nobeam_nobatch_no_ordinary_novelty_all_six_r50_20260718_v4_chtc'
PROFILE_CONTRACT_SHA256 = '81b072c03f9866817a4fc6173017788223ab8b5ba007d6015315e39d3fb4c30e'
SOURCE_ARCHIVE_SHA256 = '43a39a30d75fd3524e4f61a8339bde31f75465cf3dbbe6244f09eeaecef940f8'
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert _sha(BUNDLE_DIR / "source_locked.tar.gz") == SOURCE_ARCHIVE_SHA256
    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == BUNDLE_ID
        assert job["route_identity"]["profile_contract_sha256"] == PROFILE_CONTRACT_SHA256
        assert job["route_identity"]["profile_contract"]["execution_settings"]["phase_live_hysteresis_enabled"] is False
        assert "--phase-live-hysteresis-disabled" in job["command"]["argv"]
        assert int(job["segment"]["target_controller_round"]) == 50
    return True
if __name__ == "__main__":
    verify(); print("immutable successor verification passed")
