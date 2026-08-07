#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"fanout_bundle_receipt.json").read_text())
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 assert r["source_archive_sha256"]=='ced6b10d6bfbe4ae6a54495ff2ef4747a90036fa2027b0386555d016d5869a05'
 revision=json.loads((B/"source_revision_manifest.json").read_text())
 assert revision["profile_request"]=='sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1'
 assert revision["profile_contract_sha256"]=='9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417'
 transition=revision["source_locked_route_transition"]
 assert transition["parent_profile_contract_sha256"]=='fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2'
 assert transition["candidate_profile_contract_sha256"]=='9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417'
 expected={'weak_weak': {'u_over_t': 0.25, 'lambda': 0.25, 'g_ep': 0.353553390593, 'n_ph': 3}, 'intermediate_weak': {'u_over_t': 1.25, 'lambda': 0.25, 'g_ep': 0.353553390593, 'n_ph': 3}, 'strong_weak_u8': {'u_over_t': 8.0, 'lambda': 0.25, 'g_ep': 0.353553390593, 'n_ph': 3}, 'weak_strong': {'u_over_t': 0.25, 'lambda': 1.25, 'g_ep': 0.790569415042, 'n_ph': 7}, 'intermediate_strong': {'u_over_t': 1.25, 'lambda': 1.25, 'g_ep': 0.790569415042, 'n_ph': 7}, 'strong_strong_u8': {'u_over_t': 8.0, 'lambda': 1.25, 'g_ep': 0.790569415042, 'n_ph': 7}}
 jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
 for p in jobs:
  j=json.loads(p.read_text())
  e=expected[j["regime_slug"]]; physics=j["physics"]
  assert abs(float(physics["u_over_t"])-float(e["u_over_t"]))<=1e-12
  assert abs(float(physics["lambda"])-float(e["lambda"]))<=1e-12
  assert abs(float(physics["g_ep"])-float(e["g_ep"]))<=1e-12
  assert int(physics["n_ph_work"])==int(e["n_ph"])
  assert int(physics["n_ph_reference"])==int(e["n_ph"])
  assert physics["same_cutoff_reference"] is True
  assert j["route_identity"]["profile_request"]=='sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1'
  assert j["route_identity"]["profile_contract_sha256"]=='9d6cfae3fda84eb6a24232358c45a8f25b42c6147bfa4250905910eff687a417'
  contract=j["route_identity"]["profile_contract"]
  settings=contract["execution_settings"]
  invariants=contract["semantic_invariants"]
  assert settings["phase3_response_coordinate_scope"]=='candidate_material_coupling_window_v1'
  assert invariants["phase3_material_window_support_change_policy"]=='full_geometry_refresh_on_unexpected_supported_nullity_drift_v1'
  assert settings["phase1_prune_enabled"] is False
  assert settings["adapt_beam_live_branches"]==1
  assert int(j["segment"]["target_controller_round"])==50
  assert j["physics"]["same_cutoff_reference"] is True
 assert json.loads((B/"source_locked_sensitivity_audit.json").read_text())["fanout_authorized"] is True
 assert "requirements = False" not in (B/"submit.sub").read_text()
 return True
if __name__=="__main__": verify(); print("material-window fanout verified")
