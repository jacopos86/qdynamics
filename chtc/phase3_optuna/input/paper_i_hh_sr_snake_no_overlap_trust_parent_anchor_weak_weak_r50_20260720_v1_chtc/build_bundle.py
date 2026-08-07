#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    r=json.loads((B/"anchor_bundle_receipt.json").read_text())
    assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
    jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
    j=json.loads(jobs[0].read_text())
    assert j["route_identity"]["profile_request"]=='sr_snake_no_prune_symmetric_cost_projected_phase3_v1'
    assert j["route_identity"]["profile_contract_sha256"]=='3ff2abb1455cda3cf8cc2de0cf739172f8cdcfe6b1c9436e1afdd40076cd3ce8'
    assert j["route_identity"]["profile_contract"]["execution_settings"]["historical_singleton_trust_region_update_policy"]=="displacement_calibrated_unbounded_v2"
    assert int(j["segment"]["target_controller_round"])==50
    assert j["physics"]["n_ph_work"]==j["physics"]["n_ph_reference"]==3
    assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("no-overlap trust anchor bundle verified")
