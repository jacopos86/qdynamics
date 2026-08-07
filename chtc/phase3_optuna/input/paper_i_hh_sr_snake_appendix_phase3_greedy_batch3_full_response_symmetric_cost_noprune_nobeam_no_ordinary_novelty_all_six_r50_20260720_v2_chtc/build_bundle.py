#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, tarfile, tempfile, subprocess, sys, os
from pathlib import Path
BUNDLE = Path(__file__).resolve().parent
ROUTE = '7554bb2488a26573039eb94a74e2697b38d883a53698515a9b3ed0e5ea0fef9f'
SOURCE = 'ef3fb0ec04b5fc0242fe6b640ec4dff57c857d440cf948fa40e2196078e939cd'
PARENT_ROUTE = '27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050'
PARENT_SOURCE = 'f11607321e426d73627910a1da76a22a96f4d4bd82f66708b5b202b2e5a61453'
PROFILE = 'sr_snake_no_prune_symmetric_cost_phase3_greedy_batch_v1'
MODE = 'greedy_reduced_plane'
PATCH_SHA = '68136ab2e85b6bc73ee301675d214dfdb8944a4342e495d00b65156d66ecfea9'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    assert sha(BUNDLE / "source_locked.tar.gz") == SOURCE
    assert sha(BUNDLE / "phase3_greedy_batch_mode_overlay.patch") == PATCH_SHA
    manifest = json.loads((BUNDLE / "source_archive_manifest.json").read_text())
    assert manifest["archive_sha256"] == SOURCE
    assert manifest["greedy_batch_derivation"]["parent_source_archive_sha256"] == PARENT_SOURCE
    with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as handle:
        files = {m.name: handle.extractfile(m).read() for m in handle if m.isfile()}
    assert set(files) == set(manifest["files"])
    for name, data in files.items():
        assert hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"]
    jobs = sorted((BUNDLE / "jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        job = json.loads(path.read_text())
        assert job["bundle_id"] == 'paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_r50_20260720_v2_chtc'
        route = job["route_identity"]
        assert route["profile_request"] == PROFILE
        assert route["profile_contract_sha256"] == ROUTE
        settings = route["profile_contract"]["execution_settings"]
        assert settings["phase2_enable_batching"] is False
        assert settings["phase3_enable_batching"] is True
        assert settings["phase2_batch_selection_mode"] == MODE
        assert settings["phase3_batch_selection_mode"] == MODE
        assert settings["phase3_batch_target_size"] == 3
        assert settings["phase3_batch_size_cap"] == 3
        assert int(job["segment"]["target_controller_round"]) == 50
        assert job["source_lock"]["source_archive_sha256"] == SOURCE
    audit = json.loads((BUNDLE / "source_locked_sensitivity_audit.json").read_text())
    assert audit["status"] == "pass_exact_one_mechanism_change"
    return True
if __name__ == "__main__":
    verify(); print("immutable fixed-source greedy batch-3 verification passed")
