#!/usr/bin/env python3
import hashlib, json, tarfile
from pathlib import Path
B=Path(__file__).resolve().parent
SOURCE='8f5f88aa18a529906bbf6861d8a2411a3e3c3f0a0e9f7091fbe0a0d8afb443cc'
ROUTE='fd5ec3fa2c98b2a9d1cbcc304241d723f57dbd6210f4ea2daf30753603a146c2'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    assert h(B/"source_locked.tar.gz")==SOURCE
    jobs=sorted((B/"jobs").glob("*.json")); assert len(jobs)==6
    for path in jobs:
        job=json.loads(path.read_text())
        assert job["bundle_id"]=='paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc'
        assert job["source_lock"]["source_archive_sha256"]==SOURCE
        assert job["route_identity"]["profile_contract_sha256"]==ROUTE
        assert int(job["segment"]["target_controller_round"])==50
    with tarfile.open(B/"source_locked.tar.gz","r:gz") as archive:
        text=archive.extractfile('pipelines/static_adapt/adapt_pipeline.py').read().decode()
    marker="SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1"
    assert text.count(marker)>=3
    assert "requirements = False" not in (B/"submit.sub").read_text()
    return True
if __name__=="__main__": verify(); print("no-overlap registration successor verified")
