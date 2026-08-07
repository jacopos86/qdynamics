#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
BUNDLE_DIR = Path(__file__).resolve().parent
BUNDLE_ID = 'paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_all_six_r50_20260719_v5_chtc'
PROFILE_REQUEST = 'sr_snake_macro_only_physical_lanes_fs_prune_beam3x2_one_sided_cost_v1'
PROFILE = 'supported_whitened_adaptive_trust_full_response_one_sided_cost_fs_prune_nodamping_beam3x2_macro_only_physical_lanes_v1'
DIGEST = 'e3b9f24af40f3572063dd0d13bcca932870505870a8cd7822453b38e01bf6096'
SOURCE_SHA = '4c40399410b67b34a89f3cadeae59a0fd901c39132ff5cc746101c78e5acccd7'
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


def verify_prune_consumer_repair():
    import ast, tarfile
    predecessor = BUNDLE_DIR.parent / 'paper_i_hh_sr_snake_macro_beam3x2_fs_prune_one_sided_cost_all_six_r50_20260719_v3_chtc' / 'source_locked.tar.gz'
    assert _sha(predecessor) == '7c3ceaf5523f0c551e3c41c30e8f130f554935dba04fc6ec08ac9d48c1e4e3c9'
    with tarfile.open(predecessor, 'r:gz') as before, tarfile.open(BUNDLE_DIR / 'source_locked.tar.gz', 'r:gz') as after:
        before_files = {m.name: before.extractfile(m).read() for m in before.getmembers() if m.isfile()}
        after_files = {m.name: after.extractfile(m).read() for m in after.getmembers() if m.isfile()}
    assert before_files.keys() == after_files.keys()
    assert [p for p in before_files if before_files[p] != after_files[p]] == ['pipelines/static_adapt/adapt_pipeline.py']
    text = after_files['pipelines/static_adapt/adapt_pipeline.py'].decode('utf-8')
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sr_v4_prune_trial_branch_id')
    assert any(a.arg == 'parent_branch_id' for a in fn.args.kwonlyargs)
    assert 'parent_branch_id=getattr(' in text
    assert 'estimator_call_context' in text
    scope = {'hashlib': hashlib, 'json': json, '_SR_V4_PRUNE_TRIAL_BRANCH_PREFIX': 'sr_v4_prune_trial:'}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<archive-prune-id>', 'exec'), scope)
    branch_id = scope['_sr_v4_prune_trial_branch_id']
    shared = {'selector_step': 5, 'candidate_index': 2, 'candidate_label': 'macro:test'}
    ids = {branch_id(**shared), branch_id(**shared, parent_branch_id='beam:a'), branch_id(**shared, parent_branch_id='beam:b')}
    assert len(ids) == 3
    return True

def verify_beam_lane_serialization():
    import tarfile
    with tarfile.open(BUNDLE_DIR / 'source_locked.tar.gz', 'r:gz') as handle:
        text = handle.extractfile('pipelines/static_adapt/adapt_pipeline.py').read().decode('utf-8')
    anchor = '"selected_feature_rows": [\n                    dict(row) for row in selected_batch_feature_rows_local'
    assert text.count(anchor) == 1
    beam = text[text.index(anchor):text.index(anchor) + 8000]
    fields = ('static_lane_route', 'physical_operator_lane', 'physical_operator_quality', 'physical_operator_hh_full_meta_class', 'physical_operator_lane_source', 'physical_operator_lane_health', 'physical_operator_lane_relative_health', 'physical_operator_lane_live')
    assert all(('"' + field + '"') in beam for field in fields)
    return True

_original_verify = verify
def verify():
    assert _original_verify()
    assert verify_prune_consumer_repair()
    assert verify_beam_lane_serialization()
    return True

if __name__ == '__main__':
    verify()
    print('macro beam-prune v5 cost bundle verification passed')
