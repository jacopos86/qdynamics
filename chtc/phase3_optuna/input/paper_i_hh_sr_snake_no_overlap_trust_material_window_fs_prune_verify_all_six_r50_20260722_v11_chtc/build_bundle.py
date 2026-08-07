#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 r=json.loads((B/"fanout_bundle_receipt.json").read_text())
 assert r["bundle_id"]=='paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_all_six_r50_20260722_v11_chtc'
 assert r["batch_name"]=='paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v11'
 assert h(B/"source_locked.tar.gz")==r["source_archive_sha256"]
 repair=json.loads((B/"operational_repair.json").read_text())
 assert repair["scientific_settings_changed"] is False
 assert repair["route_contract_sha256_unchanged"]=='b43b23181ab1d93294fd2fb4ab96b32f7669c82db38082c86af39636cdf05201'
 jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
 for p in jobs:
  j=json.loads(p.read_text())
  assert j["route_identity"]["profile_contract_sha256"]=='b43b23181ab1d93294fd2fb4ab96b32f7669c82db38082c86af39636cdf05201'
  assert int(j["segment"]["target_controller_round"])==50
  assert j["physics"]["same_cutoff_reference"] is True
  assert j["source_lock"]["source_archive_sha256"]==r["source_archive_sha256"]
  c=j["route_identity"]["profile_contract"]
  assert c["semantic_invariants"]["prune_verification_beam"]=="minimal_immutable_keep_vs_one_delete_refit_sibling_v1"
  assert c["semantic_invariants"]["historical_admission_beam_active"] is False
 assert "requirements = False" not in (B/"submit.sub").read_text()
 return True
if __name__=="__main__": verify(); print("Test-2 v2 successor verified")
