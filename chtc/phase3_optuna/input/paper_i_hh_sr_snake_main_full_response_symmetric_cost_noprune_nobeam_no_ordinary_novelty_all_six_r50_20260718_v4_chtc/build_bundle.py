#!/usr/bin/env python3
"""Verify this immutable successor; it intentionally does not rebuild it."""
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_main_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260718_v4_chtc'
PROFILE_CONTRACT_SHA256 = '023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91'
SOURCE_ARCHIVE_SHA256 = '5f3e2cb2d30e49b001c72ba3c2c58fa95932da00780dc035179767df23f22041'
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
