from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc"
)
EXPECTED_IDS = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_weak_u8__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__weak_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_strong_u8__nph7__ra_singleton_plateau",
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
ROUTE_CONTRACT_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)
IMPLEMENTATION_SHA256 = (
    "1abcefba4fe1f611fc98f0392d84f40d891b16425dbd8d8bd93b2d2578e823b4"
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", *args],
        cwd=PACKAGE_DIR,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        },
        text=True,
        capture_output=True,
        check=False,
    )


def _contract_module():
    spec = importlib.util.spec_from_file_location(
        "phase3_on_plateau_r50_package_contract",
        PACKAGE_DIR / "package_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_exact_six_fresh_phase3_on_plateau_singleton_cells_validate() -> None:
    completed = _run("validate_package.py", "--deep")
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["row_count"] == 6
    assert tuple(receipt["execution_ids"]) == EXPECTED_IDS
    assert receipt["deep_preflight_count"] == 6
    assert receipt["global_source_count"] == 4
    assert receipt["execution_authorized"] is False
    assert receipt["submission_authorized"] is False

    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    assert tuple(manifest["execution_ids"]) == EXPECTED_IDS
    source_manifest = json.loads(
        (PACKAGE_DIR / manifest["source_archive_manifest"]["path"]).read_text()
    )
    assert source_manifest["implementation_source_inventory_sha256"] == (
        IMPLEMENTATION_SHA256
    )
    for execution_id in EXPECTED_IDS:
        job = json.loads(
            (PACKAGE_DIR / "jobs" / f"{execution_id}.json").read_text()
        )
        protocol = json.loads((PACKAGE_DIR / job["protocol_path"]).read_text())
        method = protocol["request"]["method"]
        settings = protocol["route_contract"]["execution_settings"]
        invariants = protocol["route_contract"]["semantic_invariants"]

        assert job["execution_mode"] == "fresh_0_to_50"
        assert job["fresh_start_contract"] == {
            "kind": "fresh_start",
            "resume_archive": None,
            "source_checkpoint": None,
        }
        assert protocol["algorithm_id"] == ALGORITHM_ID
        assert protocol["horizon"] == 50
        assert protocol["request"]["execution"] == {
            "resume": {"kind": "fresh_start"},
            "stop": {"maximum_controller_rounds": 50},
        }
        assert protocol["route_contract"]["sha256"] == ROUTE_CONTRACT_SHA256
        assert protocol["source_locks"][
            "implementation_source_inventory_sha256"
        ] == IMPLEMENTATION_SHA256

        assert method == {
            "admission": {"kind": "singleton"},
            "beam": {"kind": "off"},
            "insertion": {"kind": "plateau_commutation"},
            "pruning": {"kind": "off"},
        }
        assert settings["adapt_pool"] == "full_meta"
        assert settings["adapt_pool_class_filter_json"] is None
        assert settings["adapt_pool_label_filter_json"] is None
        assert settings["adapt_inner_optimizer"] == "POWELL"
        assert settings["adapt_maxiter"] == 200
        assert settings["adapt_seed"] == 7
        assert settings["phase2_enable_batching"] is False
        assert settings["phase3_enable_batching"] is False
        assert settings["phase3_runtime_split_max_subset_size"] == 1
        assert settings["adapt_beam_live_branches"] == 1
        assert settings["ra_active_gradient_policy"] == (
            "stationary_source_response_v1"
        )
        assert settings["ra_resource_weighting_scope"] == (
            "late_resource_weighting_v1"
        )
        assert settings["ra_phase3_population_activation_policy"] == (
            "same_round_insertion_plateau_predicate_v1"
        )
        assert settings["ra_phase3_preplateau_materialization_policy"] == (
            "phase2_winner_only_refit_geometry_v1"
        )

        assert invariants["full_meta_hva_policy"] == (
            "included_no_filters_v1"
        )
        assert invariants["active_gradient_policy"] == (
            "stationary_source_response_v1"
        )
        assert invariants["resource_weighting_scope"] == (
            "late_resource_weighting_v1"
        )
        assert invariants["plateau_prior_mean_decrease_ratio_threshold"] == (
            1.0e-4
        )
        assert invariants["plateau_threshold_comparison"] == (
            "marginal_to_prior_mean_strictly_below_v2"
        )
        assert invariants["phase3_competitive_population_activation"] == (
            "same_round_insertion_plateau_predicate_v1"
        )
        assert invariants["phase3_activation_source"] == (
            "same_round_authenticated_insertion_plateau_domain_open_v1"
        )
        assert invariants["phase3_preplateau_admission_authority"] == (
            "phase2_raw_score_top_rank_v1"
        )
        assert invariants["phase3_preplateau_materialization_policy"] == (
            "phase2_winner_only_refit_geometry_v1"
        )
        assert invariants["phase3_activation_independent_latch"] is False
        assert invariants["phase3_activation_hysteresis_active"] is False
        assert invariants["plateau_hysteresis_active"] is False
        assert invariants["pruning_active"] is False
        assert invariants["online_exact_reference_used"] is False


def test_source_archive_is_exact_v5_plus_declared_target_patch() -> None:
    contract = _contract_module()
    old_manifest = json.loads(
        (
            REPO_ROOT
            / contract.SOURCE_PACKAGE_RELATIVE
            / "source/source_archive_manifest.json"
        ).read_text()
    )
    new_manifest = json.loads(
        (PACKAGE_DIR / "source/source_archive_manifest.json").read_text()
    )
    old_rows = {row["path"]: row for row in old_manifest["members"]}
    new_rows = {row["path"]: row for row in new_manifest["members"]}
    bindings = {
        path: (before, after)
        for path, before, after in contract.SOURCE_PATCH_BINDINGS
    }
    added = {
        path for path, (before, _after) in bindings.items() if before == "<absent>"
    }
    modified = set(bindings) - added

    assert set(new_rows) == set(old_rows) | added
    assert set(old_rows) - set(new_rows) == set()
    assert {
        path
        for path in set(new_rows) & set(old_rows)
        if new_rows[path] != old_rows[path]
    } == modified
    for path, (before, after) in bindings.items():
        if before == "<absent>":
            assert path not in old_rows
        else:
            assert old_rows[path]["sha256"] == before
        assert new_rows[path]["sha256"] == after

    patch_headers = {
        line[6:]
        for line in (PACKAGE_DIR / "source_patch.diff")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("+++ b/")
    }
    assert patch_headers == set(bindings)
    assert new_manifest["member_count"] == old_manifest["member_count"] + 1
    assert new_manifest["implementation_source_inventory_sha256"] == (
        IMPLEMENTATION_SHA256
    )


def test_completed_strong_weak_anchor_binds_route_and_runtime_bytes() -> None:
    contract = _contract_module()
    result_path = REPO_ROOT / contract.TARGET_COMPLETED_RESULT_RELATIVE
    result = json.loads(result_path.read_text())
    protocol = json.loads(
        (
            PACKAGE_DIR
            / "protocols/phase3_on_plateau_r50__strong_weak_u8__"
            "nph3__ra_singleton_plateau.json"
        ).read_text()
    )
    audit = json.loads((PACKAGE_DIR / "source_lock_audit.json").read_text())

    assert _sha256(result_path) == contract.TARGET_COMPLETED_RESULT_FILE_SHA256
    assert result["protocol"]["algorithm_id"] == ALGORITHM_ID
    assert result["protocol"]["route_contract"]["sha256"] == (
        ROUTE_CONTRACT_SHA256
    )
    assert result["protocol"]["source_locks"][
        "implementation_source_inventory_sha256"
    ] == IMPLEMENTATION_SHA256
    assert protocol["route_contract"] == result["protocol"]["route_contract"]
    assert audit["completed_route_anchor"] == {
        "implementation_source_inventory_sha256": IMPLEMENTATION_SHA256,
        "path": contract.TARGET_COMPLETED_RESULT_RELATIVE.as_posix(),
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "sha256": contract.TARGET_COMPLETED_RESULT_FILE_SHA256,
    }


def test_worker_fails_closed_without_external_authorization(tmp_path: Path) -> None:
    execution_id = EXPECTED_IDS[0]
    output = tmp_path / "artifacts"
    receipt = tmp_path / "receipt.json"
    completed = _run(
        "run_cell.py",
        "--run",
        "--job",
        str(PACKAGE_DIR / "jobs" / f"{execution_id}.json"),
        "--execution-authorization",
        str(tmp_path / "absent_authorization.json"),
        "--output-dir",
        str(output),
        "--receipt",
        str(receipt),
    )
    assert completed.returncode == 2
    assert not output.exists()
    assert not receipt.exists()


def test_worker_publishes_results_across_filesystems(tmp_path: Path) -> None:
    staging = tmp_path / "temporary-filesystem" / "artifacts"
    staging.mkdir(parents=True)
    (staging / "estimator_ledger.json").write_text(
        '{"status":"passed"}\n', encoding="utf-8"
    )
    output = tmp_path / "condor-scratch" / "worker_outputs" / "artifacts"

    probe = r'''
import errno
import importlib.util
from pathlib import Path
import sys

package = Path(sys.argv[1])
staging = Path(sys.argv[2])
output = Path(sys.argv[3])
sys.path.insert(0, str(package))
spec = importlib.util.spec_from_file_location("finalizer_probe", package / "run_cell.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
real_rename = module.os.rename

def reject_original_cross_device_rename(source, destination):
    if Path(source) == staging and Path(destination) == output:
        raise OSError(errno.EXDEV, "simulated cross-device directory rename")
    return real_rename(source, destination)

module.os.rename = reject_original_cross_device_rename
module._publish_staging_directory(staging, output)
assert (output / "estimator_ledger.json").read_text(encoding="utf-8") == '{"status":"passed"}\n'
'''
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            probe,
            str(PACKAGE_DIR),
            str(staging),
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
