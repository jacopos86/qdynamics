#!/usr/bin/env python3
import hashlib, json, re
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='888270279fe1d2bceddb3ca8c0a02599dcd753d0708ce2b63118e777dbf5214d'
ROUTE='27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050'
STATE='frozen_phase3_batch3_hysteresis_disabled_v4_plus_serialized_zero_extent_matrix_receipt_repair_v5_plus_batch_selector_workspace_receipt_repair_v6+accepted_batch_coordinate_receipt_repair_v1'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v11_chtc'
        assert job["batch_name"]=='paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-noprune-nobeam-nonovelty-six-r50-20260720-v11'
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["source_lock"]["worker_source_mode"]==STATE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
    text=(B/"run_job.py").read_text()
    match=re.search(r"SOURCE_LOCK_STATE = ([^\n]+)", text); assert match
    assert eval(match.group(1), {})==STATE
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("batch3 contract-closure successor verified")
