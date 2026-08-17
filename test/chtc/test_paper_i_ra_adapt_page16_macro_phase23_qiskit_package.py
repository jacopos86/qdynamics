from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc"
)
BUNDLE_DIR = (
    PACKAGE_DIR
    / "bundle_materialization/"
    "ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_cap24_tau1em4_weak50_strong30_v1"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
BACKEND_COMPILE_SCOPE = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_page16_package_is_six_cell_inert_and_shallow_valid() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    assert manifest["status"] == "passed_inert_six_cells"
    assert manifest["row_count"] == 6
    assert manifest["weak_holstein_horizon"] == 50
    assert manifest["strong_holstein_horizon"] == 30
    assert manifest["activation_artifacts_present"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submit_descriptor_present"] is False
    assert manifest["submitted"] is False
    assert not (PACKAGE_DIR / "submit.sub").exists()

    completed = subprocess.run(
        [sys.executable, "-B", "validate_package.py"],
        cwd=PACKAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    validation = json.loads(completed.stdout)
    assert validation["status"] == "passed_inert_package"
    assert validation["shallow_worker_preflight_count"] == 6
    assert validation["launch_ready"] is False


def test_page16_protocols_keep_macros_and_use_qiskit_only_in_phases_ii_iii() -> None:
    protocols = sorted((BUNDLE_DIR / "protocols").glob("*.json"))
    assert len(protocols) == 6
    route_digests: set[str] = set()
    for path in protocols:
        protocol = _load(path)
        route = protocol["route_contract"]
        assert isinstance(route, dict)
        execution = route["execution_settings"]
        invariants = route["semantic_invariants"]
        assert isinstance(execution, dict)
        assert isinstance(invariants, dict)
        route_digests.add(str(route["sha256"]))

        assert protocol["algorithm_id"] == ALGORITHM_ID
        assert protocol["candidate_representation"] == "macro_generator_v1"
        request = protocol["request"]
        assert isinstance(request, dict)
        adapter = request["adapter"]
        assert isinstance(adapter, dict)
        assert adapter["adapter_id"] == (
            "paper_i_ra_adapt_macro_gradient_phase0_candidate_adapter_v1"
        )
        assert execution["ra_phase0_gradient_shortlist_size"] == 24
        assert execution["phase3_backend_cost_mode"] == "marrakesh_graph_span_v1"
        assert execution["phase3_backend_cost_scope"] == BACKEND_COMPILE_SCOPE
        assert execution["phase3_backend_name"] == "FakeMarrakesh"
        assert execution["phase3_backend_optimization_level"] == 1
        assert execution["phase3_backend_transpile_seed"] == 7
        assert execution["phase3_hardware_cost_normalization_mode"] == (
            "zero_centered_signed_arctan_v1"
        )
        assert execution["static_lane_route"] == "global_single_population"
        assert "physical_lane_shortlist_aggressiveness" not in execution
        assert invariants["selector_qiskit_compile_cost_active"] is True
        assert invariants["selector_compile_cost_scope"] == BACKEND_COMPILE_SCOPE
        assert invariants["selector_compile_cost_policy"] == (
            "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
        )
        assert invariants["phase_i_compile_cost_source"] == "structural_proxy_v1"
        assert invariants["phase_ii_compile_cost_source"] == "backend_transpile_v1"
        assert invariants["phase_iii_compile_cost_source"] == "backend_transpile_v1"
        assert invariants[
            "phase_ii_phase_iii_qiskit_negative_delta_reward_enabled"
        ] is True
        assert invariants["physical_operator_lanes_active"] is False
        assert invariants["macro_generator_identity_preserved_all_phases"] is True
        assert invariants["singleton_child_exposure_active"] is False
        assert "post_exposure_singleton_phase_i_policy" not in invariants
        assert invariants["candidate_funnel_order"] == (
            "macro_gradient_phase0_shortlist_then_macro_phase1_then_identity_"
            "macro_phase2_then_macro_phase3_v1"
        )
        assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == 1.0e-4
    assert len(route_digests) == 1


def test_page16_horizons_resources_and_source_lock_delta_are_exact() -> None:
    jobs = [_load(path) for path in sorted((PACKAGE_DIR / "jobs").glob("*.json"))]
    assert len(jobs) == 6
    assert {
        (int(job["nph"]), int(job["target_horizon"]))
        for job in jobs
    } == {(3, 50), (7, 30)}
    assert {
        (
            int(job["nph"]),
            int(job["resources"]["request_cpus"]),
            int(job["resources"]["request_memory_mb"]),
            int(job["resources"]["request_disk_mb"]),
        )
        for job in jobs
        if isinstance(job["resources"], dict)
    } == {
        (3, 4, 32_768, 61_440),
        (7, 4, 49_152, 81_920),
    }

    audit = _load(PACKAGE_DIR / "source_lock_audit.json")
    parent = audit["mechanical_control_plane_parent"]
    assert isinstance(parent, dict)
    assert str(parent["path"]).endswith(
        "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
        "cap24_tau1em4_r50_20260810_v3_chtc/package_manifest.json"
    )
    changes = audit["approved_settings_changes_from_page13_v3"]
    assert isinstance(changes, dict)
    assert set(changes) == {
        "selector_cost_route",
        "strong_holstein_horizon",
    }
    assert audit["non_requested_scientific_settings_diff"] == []

    source_manifest = _load(PACKAGE_DIR / "source/source_archive_manifest.json")
    members = source_manifest["members"]
    assert isinstance(members, list)
    paths = {
        str(row["path"])
        for row in members
        if isinstance(row, dict)
    }
    assert {
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/hh_backend_compile_oracle.py",
        "pipelines/static_adapt/ra_adapt/adapters.py",
        "pipelines/static_adapt/ra_adapt/engine.py",
        "pipelines/static_adapt/ra_adapt/phase0.py",
    } <= paths
