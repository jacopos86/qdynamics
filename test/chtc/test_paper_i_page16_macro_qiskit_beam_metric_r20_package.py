from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_20260812_"
    "v2_chtc"
)
BATCH_NAME = "paper-i-page16-macro-qiskit-beam3x2-metric-r20-20260812-v2"
ROUTE_SHA256 = (
    "62dd2b102d7b664121c9265e1b7e2e97382d2acb8fdcfe7238ad9ae28720d452"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_package_is_six_cell_r20_inert_and_shallow_valid() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    assert manifest["status"] == "passed_inert_six_cells"
    assert manifest["row_count"] == 6
    assert manifest["target_horizon"] == 20
    assert manifest["batch_name"] == BATCH_NAME
    assert manifest["child_route_contract_sha256"] == ROUTE_SHA256
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    validation = json.loads(completed.stdout)
    assert validation["status"] == "passed_inert_package"
    assert validation["shallow_worker_preflight_count"] == 6
    assert validation["launch_ready"] is False


def test_submit_template_renders_one_package_bound_batch_name() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "from activate_package import _render_submit_template; "
                "print(_render_submit_template("
                "package_relative='chtc/package', "
                "activation_relative='chtc/activation', "
                "source_archive_sha256='a' * 64))"
            ),
        ],
        cwd=PACKAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    expected = f'+JobBatchName = "{BATCH_NAME}"'
    assert completed.stdout.count(expected) == 1
    assert "__BATCH_NAME__" not in completed.stdout


def test_validator_rejects_undeclared_files_and_python_caches(
    tmp_path: Path,
) -> None:
    for name, relative, expected_error in (
        ("extra", "undeclared.txt", "Package membership drifted"),
        (
            "cache",
            "__pycache__/sentinel.cpython-312.pyc",
            "forbidden Python cache artifacts",
        ),
    ):
        copied = tmp_path / name
        shutil.copytree(PACKAGE_DIR, copied)
        extra = copied / relative
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_bytes(b"not declared\n")
        completed = subprocess.run(
            [sys.executable, "-B", "validate_package.py"],
            cwd=copied,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert completed.returncode == 2
        assert expected_error in completed.stderr


def test_protocols_compose_page16_qiskit_with_beam_and_metric_pruning() -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    protocols = [
        _load(PACKAGE_DIR / str(row["path"]))
        for row in manifest["protocols"]
        if isinstance(row, dict)
    ]
    assert len(protocols) == 6
    for protocol in protocols:
        assert protocol["horizon"] == 20
        assert protocol["candidate_representation"] == "macro_generator_v1"
        request = protocol["request"]
        route = protocol["route_contract"]
        assert isinstance(request, dict)
        assert isinstance(route, dict)
        method = request["method"]
        execution = route["execution_settings"]
        invariants = route["semantic_invariants"]
        lineage = route["lineage_authority"]
        assert isinstance(method, dict)
        assert isinstance(execution, dict)
        assert isinstance(invariants, dict)
        assert isinstance(lineage, dict)

        assert route["sha256"] == ROUTE_SHA256
        assert method["admission"] == {"kind": "singleton"}
        assert method["insertion"] == {"kind": "plateau_commutation"}
        assert method["pruning"] == {"kind": "metric"}
        assert method["beam"] == {
            "kind": "fork_local",
            "live_parent_branches": 3,
            "admission_children_per_parent": 2,
            "maximum_admission_children_per_round": 6,
            "s_alg_weight": 0.005,
            "calibration_status": "uncalibrated_default",
        }
        assert execution["ra_phase0_gradient_shortlist_size"] == 24
        assert execution["phase1_prune_enabled"] is True
        assert execution["phase3_backend_cost_scope"] == (
            "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
        )
        assert execution["phase3_hardware_cost_normalization_mode"] == (
            "zero_centered_signed_arctan_v1"
        )
        assert execution["static_lane_route"] == "global_single_population"
        assert "physical_lane_shortlist_aggressiveness" not in execution
        assert invariants["selector_qiskit_compile_cost_active"] is True
        assert invariants["phase_i_compile_cost_source"] == "structural_proxy_v1"
        assert invariants["phase_ii_compile_cost_source"] == "backend_transpile_v1"
        assert invariants["phase_iii_compile_cost_source"] == "backend_transpile_v1"
        assert invariants["physical_operator_lanes_active"] is False
        assert invariants["beam_shape"] == (
            "three_live_two_children_per_parent_v1"
        )
        assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == 1.0e-4
        assert invariants["macro_generator_identity_preserved_all_phases"] is True
        assert invariants["singleton_child_exposure_active"] is False
        assert lineage["parent_contract_sha256"] == (
            "1cebfef5b79ed86fc40072f896f6921da202c004e09025750e86e130141154eb"
        )


def test_queue_resources_and_page16_delta_are_closed() -> None:
    jobs = [_load(path) for path in sorted((PACKAGE_DIR / "jobs").glob("*.json"))]
    assert len(jobs) == 6
    assert {
        (
            int(job["nph"]),
            int(job["target_horizon"]),
            int(job["resources"]["request_cpus"]),
            int(job["resources"]["request_memory_mb"]),
            int(job["resources"]["request_disk_mb"]),
        )
        for job in jobs
    } == {
        (3, 20, 4, 49_152, 81_920),
        (7, 20, 4, 65_536, 102_400),
    }
    queue = [
        row.split("\t")
        for row in (PACKAGE_DIR / "queue.tsv").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    assert len(queue) == 6
    assert all(len(row) == 8 for row in queue)

    audit = _load(PACKAGE_DIR / "source_lock_audit.json")
    changes = audit["approved_settings_changes_from_page16"]
    assert isinstance(changes, dict)
    assert set(changes) == {
        "pruning",
        "beam",
        "target_horizon",
        "scheduler_resources",
    }
    assert audit["non_requested_scientific_settings_diff"] == []
    parent = audit["mechanical_control_plane_parent"]
    assert isinstance(parent, dict)
    assert str(parent["path"]).endswith(
        "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
        "phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc/"
        "package_manifest.json"
    )
