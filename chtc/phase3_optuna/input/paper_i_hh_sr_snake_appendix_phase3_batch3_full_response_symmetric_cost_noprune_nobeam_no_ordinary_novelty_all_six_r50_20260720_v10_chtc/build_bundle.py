#!/usr/bin/env python3
import hashlib, json, tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='888270279fe1d2bceddb3ca8c0a02599dcd753d0708ce2b63118e777dbf5214d'
ROUTE='27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v10_chtc'
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
        assert int(job["segment"]["target_controller_round"])==50
    with tarfile.open(B/"source_locked.tar.gz","r:gz") as t:
        text=t.extractfile('pipelines/static_adapt/adapt_pipeline.py').read().decode()
    assert "certified_positions" in text
    assert "normalize_serialized_matrix_payload" in text
    assert "commit order disagrees with the certified" in text
    assert "len(positions_in_commit_order) != int(G_BB.shape[0])" not in text
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("batch3 coordinate successor verified")
