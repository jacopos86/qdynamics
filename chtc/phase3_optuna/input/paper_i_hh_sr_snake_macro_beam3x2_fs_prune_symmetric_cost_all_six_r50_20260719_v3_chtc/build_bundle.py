#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_macro_beam3x2_fs_prune_symmetric_cost_all_six_r50_20260719_v3_chtc'
PROFILE_REQUEST = 'sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_v1'
PROFILE = 'supported_whitened_adaptive_trust_full_response_symmetric_cost_fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1'
DIGEST = 'a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0'
SOURCE_SHA = '7c3ceaf5523f0c551e3c41c30e8f130f554935dba04fc6ec08ac9d48c1e4e3c9'
COST = 'family_robust_symmetric_arctan_v1'
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
