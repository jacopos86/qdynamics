#!/usr/bin/env python3
"""Verify this immutable successor; it intentionally does not rebuild it."""
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc'
PROFILE_CONTRACT_SHA256 = '27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050'
SOURCE_ARCHIVE_SHA256 = 'e93b75e0cf9961d78f1ec9b41108a93deb9f5c039e15ee5371187ff7b103f299'
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
