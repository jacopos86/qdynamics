from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "cap24_tau1em4_r50_20260810_v2_chtc"
)
MATCHMAKING_REPAIR_PACKAGE_DIR = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "cap24_tau1em4_r50_20260810_v3_chtc"
)
BUNDLE_DIR = (
    PACKAGE_DIR
    / "bundle_materialization/"
    "ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_cap24_"
    "tau1em4_r50_v2"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "proxy_plateau_no_lanes_v1"
)
ROUTE_SHA256 = (
    "1b2f7254a96a27a7f2a262f1b4bc19c886b421a9cbaa5e24c95e354a02f2cf45"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_pinned_image_probe_replaces_preloaded_ambient_pipeline_modules(
    tmp_path: Path,
) -> None:
    ambient = tmp_path / "ambient"
    stale = ambient / "pipelines/scaffold/hh_continuation_scoring.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("STALE_AMBIENT_MODULE = True\n", encoding="utf-8")
    script = textwrap.dedent(
        """
        import importlib.util
        import json
        from pathlib import Path
        import sys

        ambient = Path(sys.argv[1]).resolve()
        probe_path = Path(sys.argv[2]).resolve()
        archive_path = Path(sys.argv[3]).resolve()
        sys.path.insert(0, ambient.as_posix())
        import pipelines.scaffold.hh_continuation_scoring as stale
        assert stale.STALE_AMBIENT_MODULE is True
        assert not hasattr(stale, "BATCH_ADDITIVITY_HARD_GATE_LEGACY_V1")

        spec = importlib.util.spec_from_file_location("sealed_probe", probe_path)
        assert spec is not None and spec.loader is not None
        probe_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe_module)
        result = probe_module.probe(archive_path)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        """
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            script,
            ambient.as_posix(),
            (PACKAGE_DIR / "probe_image_runtime.py").as_posix(),
            (PACKAGE_DIR / "source/source_locked.tar.gz").as_posix(),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["sealed_module_paths_verified"] is True
    assert payload["source_cwd_isolated"] is True
    assert payload["source_activation_policy"] == (
        "chdir_source_then_purge_ambient_modules_and_paths_before_sealed_import_v2"
    )
    paths = payload["sealed_module_paths"]
    assert isinstance(paths, dict)
    assert {
        "pipelines.scaffold.hh_continuation_scoring",
        "pipelines.static_adapt.hh_backend_compile_oracle",
        "pipelines.static_adapt.ra_adapt.adapters",
        "pipelines.static_adapt.ra_adapt.engine",
    } <= set(paths)
    assert payload["loaded_source_module_count"] == len(paths)
    for path_or_paths in paths.values():
        path_rows = (
            path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        )
        assert path_rows
        assert all(
            str(path) in {"pipelines", "src"}
            or str(path).startswith(("pipelines/", "src/"))
            for path in path_rows
        )


def test_sealed_macro_phase01_package_is_six_cell_inert_and_valid() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    assert manifest["status"] == "passed_inert_six_cells"
    assert manifest["row_count"] == 6
    assert manifest["child_route_contract_sha256"] == ROUTE_SHA256
    assert manifest["activation_artifacts_present"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submit_descriptor_present"] is False
    assert manifest["submitted"] is False
    assert not (PACKAGE_DIR / "submit.sub").exists()

    completed = subprocess.run(
        [sys.executable, "validate_package.py"],
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


def test_all_protocols_preserve_macros_and_disable_qiskit_and_lanes() -> None:
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
        assert execution["phase3_backend_cost_mode"] == (
            "marrakesh_graph_span_v1"
        )
        assert "phase3_backend_cost_scope" not in execution
        assert execution["static_lane_route"] == "global_single_population"
        assert "physical_lane_shortlist_aggressiveness" not in execution
        assert invariants["selector_qiskit_compile_cost_active"] is False
        assert invariants["physical_operator_lanes_active"] is False
        assert invariants["shortlist_population_policy"] == (
            "single_global_population_v1"
        )
        assert invariants["macro_generator_identity_preserved_all_phases"] is True
        assert invariants["singleton_child_exposure_active"] is False
        assert invariants["phase_i_phase_ii_phase_iii_cost_source"] == (
            "marrakesh_graph_span_structural_proxy_v1"
        )
        assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == (
            1.0e-4
        )
        assert "selector_compile_cost_policy" not in invariants
        assert "phase_ii_compile_cost_source" not in invariants
        assert "phase_iii_compile_cost_source" not in invariants
    assert route_digests == {ROUTE_SHA256}


def test_queue_resources_and_source_archive_match_macro_envelopes() -> None:
    rows = [line.split("\t") for line in (PACKAGE_DIR / "queue.tsv").read_text().splitlines()]
    assert len(rows) == 6
    assert {tuple(row[4:8]) for row in rows} == {
        ("4", "49152", "61440", "259200"),
        ("4", "65536", "81920", "259200"),
    }

    source_manifest = _load(PACKAGE_DIR / "source/source_archive_manifest.json")
    members = source_manifest["members"]
    assert isinstance(members, list)
    paths = {
        str(row["path"])
        for row in members
        if isinstance(row, dict)
    }
    required = {
        "pipelines/scaffold/hh_continuation_scoring.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/hh_backend_compile_oracle.py",
        "pipelines/static_adapt/ra_adapt/adapters.py",
        "pipelines/static_adapt/ra_adapt/bundles.py",
        "pipelines/static_adapt/ra_adapt/contracts.py",
        "pipelines/static_adapt/ra_adapt/engine.py",
        "pipelines/static_adapt/ra_adapt/phase0.py",
        "pipelines/static_adapt/ra_adapt/pools.py",
    }
    assert required <= paths
    assert source_manifest["member_count"] == len(members)


def test_source_audit_uses_v13_macro_science_not_page11_singleton_science() -> None:
    audit = _load(PACKAGE_DIR / "source_lock_audit.json")
    assert audit["source_route_contract_sha256"] == (
        "e7b17287fb21adf703101f44da31cdf4e716d0752600aa36dd30691384d8fbd7"
    )
    assert audit["target_route_contract_sha256"] == ROUTE_SHA256
    source_protocols = audit["scientific_source_protocols"]
    assert isinstance(source_protocols, list)
    assert len(source_protocols) == 6
    assert all(
        "ra_adapt_stationary_late_core_v13" in str(row["path"])
        and str(row["path"]).endswith("ra_macro_plateau.json")
        for row in source_protocols
        if isinstance(row, dict)
    )


def test_v3_is_a_resource_only_matchmaking_repair() -> None:
    v2_manifest = _load(PACKAGE_DIR / "package_manifest.json")
    v3_manifest = _load(MATCHMAKING_REPAIR_PACKAGE_DIR / "package_manifest.json")
    assert v3_manifest["status"] == "passed_inert_six_cells"
    assert v3_manifest["row_count"] == 6
    assert v3_manifest["child_route_contract_sha256"] == ROUTE_SHA256
    assert v3_manifest["child_route_contract_sha256"] == (
        v2_manifest["child_route_contract_sha256"]
    )
    assert v3_manifest["source_archive"]["sha256"] == (
        v2_manifest["source_archive"]["sha256"]
    )

    v2_jobs = {
        path.name: _load(path)
        for path in (PACKAGE_DIR / "jobs").glob("*.json")
    }
    v3_jobs = {
        path.name: _load(path)
        for path in (MATCHMAKING_REPAIR_PACKAGE_DIR / "jobs").glob("*.json")
    }
    assert set(v3_jobs) == set(v2_jobs)
    for name, v3_job in v3_jobs.items():
        v2_job = v2_jobs[name]
        assert v3_job["route_contract_sha256"] == v2_job["route_contract_sha256"]
        for field in (
            "active_gradient_policy",
            "algorithm_id",
            "candidate_adapter_id",
            "candidate_representation",
            "resource_weighting_scope",
            "route_id",
            "selector_qiskit_compile_cost_active",
            "structural_proxy_cost_source",
            "structural_proxy_mode",
            "target_horizon",
        ):
            assert v3_job[field] == v2_job[field]
        resources = v3_job["resources"]
        assert isinstance(resources, dict)
        expected_memory = 32_768 if int(v3_job["nph"]) == 3 else 49_152
        assert resources["request_memory_mb"] == expected_memory

    completed = subprocess.run(
        [sys.executable, "validate_package.py"],
        cwd=MATCHMAKING_REPAIR_PACKAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    validation = json.loads(completed.stdout)
    assert validation["status"] == "passed_inert_package"
    assert validation["shallow_worker_preflight_count"] == 6
