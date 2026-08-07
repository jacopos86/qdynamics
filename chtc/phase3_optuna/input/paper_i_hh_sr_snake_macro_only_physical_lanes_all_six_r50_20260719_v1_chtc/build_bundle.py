#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_macro_only_physical_lanes_all_six_r50_20260719_v1_chtc'
PROFILE = 'supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_macro_only_physical_lanes_v1'
DIGEST = 'd14d582e532ee41500cd7d3ebaa21b83da91bb3fcf014be53ab8d1049d1452fa'
SOURCE_SHA = '3a5ed36ebdf260357aa86b3a5ab3c7d8372072329a8fec2e1043e90b6f7c34c7'
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert _sha(BUNDLE_DIR / "source_locked.tar.gz") == SOURCE_SHA
    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    assert len(jobs) == 6
    assert len((BUNDLE_DIR / "queue.tsv").read_text().strip().splitlines()) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == BUNDLE_ID
        assert job["route_identity"]["profile_resolved"] == PROFILE
        assert job["route_identity"]["profile_contract_sha256"] == DIGEST
        assert int(job["segment"]["target_controller_round"]) == 50
        assert int(job["physics"]["n_ph_work"]) == int(job["physics"]["n_ph_reference"])
        assert "--phase-live-hysteresis-disabled" in job["command"]["argv"]
    return True
if __name__ == "__main__": verify(); print("pool-complement bundle verification passed")
