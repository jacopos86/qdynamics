#!/usr/bin/env python3
import hashlib, json, re
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='14a1fd9617634a8ae2ca1b1d5f3971ffe8d74bf4203ed3226ad54de72281771f'
ROUTE='ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865'
STATE='frozen_phase3_batch3_v6_plus_fixed_source_greedy_selection_v1+accepted_batch_coordinate_receipt_repair_v1'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v8_chtc'
        assert job["batch_name"]=='paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-noprune-nobeam-nonovelty-six-r50-20260720-v8'
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["source_lock"]["worker_source_mode"]==STATE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
    text=(B/"run_job.py").read_text()
    match=re.search(r"SOURCE_LOCK_STATE = ([^\n]+)", text); assert match
    assert eval(match.group(1), {})==STATE
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("batch3 contract-closure successor verified")
