#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_all_six_r50_20260719_v2_chtc'
PROFILE_REQUEST = 'sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1'
PROFILE = 'supported_whitened_adaptive_trust_full_response_one_sided_cost_fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1'
DIGEST = 'e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096'
SOURCE_SHA = '69d15c1bc7f2b704d8bf658696a551b5a15a34c2140971c6c382f0cae385cec2'
COST = 'family_robust_v1'
def _sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    assert _sha(BUNDLE_DIR / "source_locked.tar.gz") == SOURCE_SHA
    jobs = sorted((BUNDLE_DIR / "jobs").glob("*.json"))
    normalized = sorted((BUNDLE_DIR / "normalized_manifests").glob("*.json"))
    assert len(jobs) == len(normalized) == 6
    assert len((BUNDLE_DIR / "queue.tsv").read_text().strip().splitlines()) == 6
    for path in jobs + normalized:
        job = json.loads(path.read_text())
        route = job["route_identity"]
        settings = route["profile_contract"]["execution_settings"]
        semantics = route["profile_contract"]["semantic_invariants"]
        assert job["bundle_id"] == BUNDLE_ID
        assert route["profile_request"] == PROFILE_REQUEST
        assert route["profile_resolved"] == PROFILE
        assert route["profile_contract_sha256"] == DIGEST
        assert settings["adapt_beam_live_branches"] == 3
        assert settings["adapt_beam_children_per_parent"] == 2
        assert settings["phase1_prune_enabled"] is True
        assert settings["phase1_prune_metric_schur_mu"] == 0.0
        assert settings["phase1_prune_recovery_trust_radius"] == 0.125
        assert settings["phase3_hardware_cost_normalization_mode"] == COST
        assert settings["phase3_runtime_split_mode"] == "off"
        assert semantics["generated_pauli_children_active"] is False
        assert semantics["physical_operator_lanes_active"] is True
        assert semantics["pruning_active"] is True
        assert int(job["segment"]["target_controller_round"]) == 50
        assert int(job["physics"]["n_ph_work"]) == int(job["physics"]["n_ph_reference"])
        argv = job.get("command", {}).get("argv") or job.get("command_argv", [])
        assert "--phase-live-hysteresis-disabled" in argv
    return True
if __name__ == "__main__": verify(); print("macro beam-prune cost bundle verification passed")
