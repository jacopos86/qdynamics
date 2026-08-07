#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"anchor_bundle_receipt.json").read_text())
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 jobs=list((B/"jobs").glob("*.json")); assert [p.name for p in jobs]==["weak_weak.json"]
 j=json.loads(jobs[0].read_text())
 assert j["route_identity"]["profile_request"]=='sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1'
 assert j["route_identity"]["profile_contract_sha256"]=='fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2'
 assert j["route_identity"]["profile_contract"]["execution_settings"]["phase3_response_coordinate_scope"]=='full_active_plus_singleton_v1'
 assert int(j["segment"]["target_controller_round"])==50
 assert j["physics"]["n_ph_work"]==j["physics"]["n_ph_reference"]==3
 assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is False
 assert "requirements = False" not in (B/"submit.sub").read_text()
 return True
if __name__=="__main__": verify(); print("material-window parent anchor verified")
