#!/usr/bin/env python3
import hashlib, json, tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='5b5fd8d1f478778ecbee612ae3b0cb61da9267851ad90c4c5add6fe77e8b4789'
ROUTE='ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v9_chtc'
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
