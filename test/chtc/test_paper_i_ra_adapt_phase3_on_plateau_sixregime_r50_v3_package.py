from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
V2_PACKAGE = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc"
)
V3_PACKAGE = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc"
)
V2_CANARY = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_canary_weak_strong_v1"
)
V3_CANARY = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc_activation_canary_weak_strong_v1"
)
V2_REMAINING = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_remaining5_v1"
)
V3_REMAINING = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc_activation_remaining5_v1"
)
EXPECTED_IDS = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_weak_u8__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__weak_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_strong_u8__nph7__ra_singleton_plateau",
)
CANARY_ID = EXPECTED_IDS[3]
OPERATIONAL_CHANGE = (
    "authenticated_geometry_expansion_null_phase3_stabilization_v1"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _run(path: Path, script: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-B", script],
        cwd=path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_v3_is_exact_v2_source_plus_reviewed_engine_repair() -> None:
    contract = _load_module(
        V3_PACKAGE / "package_contract.py",
        "phase3_on_plateau_v3_package_contract",
    )
    before_manifest = json.loads(
        (V2_PACKAGE / "source/source_archive_manifest.json").read_text()
    )
    after_manifest = json.loads(
        (V3_PACKAGE / "source/source_archive_manifest.json").read_text()
    )
    before = {row["path"]: row for row in before_manifest["members"]}
    after = {row["path"]: row for row in after_manifest["members"]}
    expected_patch = {
        path: (before_sha, after_sha)
        for path, before_sha, after_sha in contract.SOURCE_PATCH_BINDINGS
    }

    assert set(expected_patch) == {
        "pipelines/static_adapt/ra_adapt/engine.py"
    }
    assert set(after) == set(before)
    assert {path for path in after if after[path] != before[path]} == set(
        expected_patch
    )
    for path, (before_sha, after_sha) in expected_patch.items():
        assert before[path]["sha256"] == before_sha
        assert after[path]["sha256"] == after_sha
    assert after_manifest["member_count"] == before_manifest["member_count"]
    assert after_manifest["implementation_source_inventory_sha256"] == (
        contract.TARGET_IMPLEMENTATION_INVENTORY_SHA256
    )

    locks = json.loads((V3_PACKAGE / "source_locks_snapshot.json").read_text())
    audit = json.loads((V3_PACKAGE / "source_lock_audit.json").read_text())
    plan = json.loads((V3_PACKAGE / "execution_plan.json").read_text())
    assert locks["sha256"] == contract.TARGET_SOURCE_LOCKS_CANONICAL_SHA256
    assert locks["implementation_sources"]["sha256"] == (
        contract.TARGET_IMPLEMENTATION_INVENTORY_SHA256
    )
    assert audit["scientific_changes"] == []
    assert audit["operational_changes"] == [OPERATIONAL_CHANGE]
    assert plan["worker_owned_live_progress"] is True
    assert plan["checkpoint_dynamic_sidecars_hash_bound"] is True
    assert plan["same_filesystem_atomic_success_publication"] is True
    assert plan["post_validation_failure_evidence_preserved"] is True


def test_v3_preserves_every_v2_scientific_protocol_field() -> None:
    contract = _load_module(
        V3_PACKAGE / "package_contract.py",
        "phase3_on_plateau_v3_protocol_contract",
    )
    expected_differences = set(contract.SOURCE_TO_TARGET_DIFFERENCE_PATHS)
    observed_ids: list[str] = []
    for after_path in sorted((V3_PACKAGE / "protocols").glob("*.json")):
        before = json.loads(
            (V2_PACKAGE / "protocols" / after_path.name).read_text()
        )
        after = json.loads(after_path.read_text())
        differences = contract.scalar_differences(before, after)
        assert {path for path, _before, _after in differences} == (
            expected_differences
        )
        for key in (
            "problem",
            "parent_inventory",
            "executable_pool",
            "optimizer",
            "optimizer_maxiter",
            "seeds",
            "candidate_representation",
            "active_gradient_policy",
            "resource_weighting_scope",
            "route_contract",
            "request",
        ):
            assert after[key] == before[key]
        observed_ids.append(after["bundle_materialization"]["cell_id"])
    assert tuple(observed_ids) == tuple(sorted(EXPECTED_IDS))

    for execution_id in EXPECTED_IDS:
        before_job = json.loads(
            (V2_PACKAGE / "jobs" / f"{execution_id}.json").read_text()
        )
        after_job = json.loads(
            (V3_PACKAGE / "jobs" / f"{execution_id}.json").read_text()
        )
        for key in (
            "active_gradient_policy",
            "candidate_representation",
            "exact_same_cutoff_energy",
            "execution_mode",
            "fresh_start_contract",
            "insertion_policy",
            "plateau_prior_mean_decrease_ratio_threshold",
            "plateau_threshold_comparison",
            "plateau_trigger_source",
            "resource_weighting_scope",
            "resources",
            "route_contract_sha256",
            "target_horizon",
        ):
            assert after_job[key] == before_job[key]


def test_v3_preserves_v2_worker_durability_bytes() -> None:
    assert (V3_PACKAGE / "run_cell.py").read_bytes() == (
        V2_PACKAGE / "run_cell.py"
    ).read_bytes()
    assert (V3_PACKAGE / "execute_authorized_job.sh").read_bytes() == (
        V2_PACKAGE / "execute_authorized_job.sh"
    ).read_bytes()
    for before, after in (
        (V2_CANARY, V3_CANARY),
        (V2_REMAINING, V3_REMAINING),
    ):
        assert (after / "build_attempt_archive.py").read_bytes() == (
            before / "build_attempt_archive.py"
        ).read_bytes()
        assert (after / "execute_authorized_job.sh").read_bytes() == (
            before / "execute_authorized_job.sh"
        ).read_bytes()


def test_v3_package_and_split_activations_validate_closed() -> None:
    package_receipt = _run(V3_PACKAGE, "validate_package.py")
    canary_receipt = _run(V3_CANARY, "validate_activation.py")
    remaining_receipt = _run(V3_REMAINING, "validate_activation.py")
    package_manifest = json.loads(
        (V3_PACKAGE / "package_manifest.json").read_text()
    )
    canary = json.loads((V3_CANARY / "activation_manifest.json").read_text())
    remaining = json.loads(
        (V3_REMAINING / "activation_manifest.json").read_text()
    )

    assert package_receipt["status"] == "passed"
    assert package_receipt["execution_ids"] == list(EXPECTED_IDS)
    assert canary_receipt["status"] == "passed"
    assert remaining_receipt["status"] == "passed"
    assert canary_receipt["ordinary_held"] is False
    assert remaining_receipt["ordinary_held"] is False
    assert canary_receipt["factory"] is False
    assert remaining_receipt["factory"] is False
    assert canary["direct_execution_count"] == 1
    assert remaining["direct_execution_count"] == 5
    canary_ids = {row["execution_id"] for row in canary["executions"]}
    remaining_ids = {row["execution_id"] for row in remaining["executions"]}
    assert canary_ids == {CANARY_ID}
    assert remaining_ids == set(EXPECTED_IDS) - canary_ids
    assert canary_ids.isdisjoint(remaining_ids)
    assert canary_ids | remaining_ids == set(EXPECTED_IDS)
    for activation in (canary, remaining):
        assert activation["sealed_package"]["manifest"][
            "canonical_sha256"
        ] == package_manifest["sha256"]
        assert activation["remote_stage"] is False
        assert activation["condor_submit"] is False
        assert activation["submitted"] is False
        assert activation["submission_state"] == (
            "authorized_pending_remote_preflight"
        )
    assert canary["activation_id"] != remaining["activation_id"]
    assert canary["batch_name"] != remaining["batch_name"]
    assert "runtime_canary_weak_strong_v1" in (
        V3_CANARY / "submit.sub"
    ).read_text()
    assert "runtime_remaining5_v1" in (
        V3_REMAINING / "submit.sub"
    ).read_text()
