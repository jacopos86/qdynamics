#!/usr/bin/env python3
import hashlib, json, tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='31de9ad4d42c75a8dd0e95815144dc93bfaafefc30f073dba97c0bf69591761b'
ROUTE='27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v17_chtc'
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
        assert int(job["segment"]["target_controller_round"])==50
    with tarfile.open(B/"source_locked.tar.gz","r:gz") as t:
        text=t.extractfile('pipelines/static_adapt/adapt_pipeline.py').read().decode()
    assert "def _restore_phase3_batch_singleton_coordinate_receipts(" in text
    assert "authoritative_full_response_admission_record_v1" in text
    assert "raw Phase-II fallback is forbidden" in text
    assert "phase3_batch_singleton_coordinate_receipt_restoration_v1" in text
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("batch3 coordinate successor verified")
