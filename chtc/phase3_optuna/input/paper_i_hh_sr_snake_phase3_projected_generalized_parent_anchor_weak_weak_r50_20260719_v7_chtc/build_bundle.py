#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
B = Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    receipt=json.loads((B/"anchor_bundle_receipt.json").read_text())
    assert h(B/"source_locked.tar.gz")==receipt["source_archive_sha256"]
    assert not (B/"SCIENTIFIC_BLOCKER_DO_NOT_SUBMIT.json").exists()
    assert json.loads((B/"submission_gate.json").read_text())["scientific_blockers"]==[]
    assert "bash -lc 'set -euo pipefail; cd /work" in (B/"execute_source_locked_job.sh").read_text()
    jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
    job=json.loads(jobs[0].read_text())
    assert job["route_identity"]["profile_contract_sha256"]=='023bc7ac535ee4d88d78dd5336a59dd2fb0543c133fa0a60b009efab75422c91'
    assert job["route_identity"]["profile_contract"]["execution_settings"]["historical_singleton_coordinate_solve_policy"]=="supported_metric_whitened_eigh_v1"
    assert int(job["segment"]["target_controller_round"])==50
    assert job["physics"]["n_ph_work"]==job["physics"]["n_ph_reference"]==3
    assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
    return True
if __name__=="__main__": verify(); print("anchor bundle verification passed")
