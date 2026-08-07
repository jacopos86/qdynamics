from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATION_ROOT = (
    REPAIR_ROOT
    / "bundles/materializations/ra_adapt_global_singleton_insertion12_v1"
)
PACKAGE_DIR = (
    REPAIR_ROOT
    / "paper_i_ra_adapt_global_singleton_insertion12_r50_20260730_v1_chtc"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_global_singleton_insertion12_contract",
        path,
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


def test_materialization_is_exact_inert_and_source_locked() -> None:
    contract = _load_contract()
    authority = contract.validate_materialization_authority(
        REPO_ROOT
    )
    final = authority["final"]
    assert final["status"] == "passed"
    assert final["run_class"] == "diagnostic"
    assert final["cell_count"] == 12
    assert final["execution_authorized"] is False
    assert final["submission_authorized"] is False
    assert final["submitted"] is False
    assert authority["equality_audit"]["status"] == "passed"
    assert authority["equality_audit"]["allowed_axis"] == (
        "insertion_policy"
    )
    assert authority["equality_audit"]["regime_pair_count"] == 6
    assert authority["equality_audit"]["variant_count"] == 12
    delta = authority["source_lock_delta"]
    assert delta["status"] == "passed"
    assert delta["source_cell_count"] == 6
    assert delta["derived_cell_count"] == 12
    assert delta["all_archive_bindings_preserved"] is True
    assert delta["all_member_bindings_preserved"] is True
    assert delta["all_global_source_bindings_preserved"] is True
    for paths in authority["artifact_destinations"].values():
        assert len(paths) == 12
        assert len(set(paths)) == 12


def test_rows_cover_six_regimes_by_two_insertion_policies() -> None:
    contract = _load_contract()
    authority = contract.validate_materialization_authority(
        REPO_ROOT
    )
    rows = list(contract.direct_execution_rows())
    assert len(rows) == 12
    assert len({row["execution_id"] for row in rows}) == 12
    assert {
        (row["regime_id"], row["nph"], row["route_id"])
        for row in rows
    } == {
        (regime_id, nph, route_id)
        for regime_id, nph in contract.REGIME_CUTOFF_PAIRS
        for route_id in contract.ROUTE_IDS
    }
    assert all(
        row["candidate_adapter_id"]
        == (
            "paper_i_ra_adapt_global_single_pauli_word_"
            "candidate_adapter_v1"
        )
        for row in rows
    )
    assert all(
        row["active_gradient_policy"]
        == "stationary_source_response_v1"
        and row["resource_weighting_scope"]
        == "all_phase_resource_weighting_v1"
        and row["phase1_cost_term"] == "enabled"
        for row in rows
    )

    for row in rows:
        protocol = _load(
            REPO_ROOT
            / authority["protocol_bindings"][
                row["execution_id"]
            ]["path"]
        )
        assert protocol["request"]["adapter"]["adapter_id"] == (
            row["candidate_adapter_id"]
        )
        assert protocol["request"]["method"]["insertion"]["kind"] == (
            row["insertion_policy"]
        )
        assert protocol["route_contract"]["execution_settings"][
            "adapt_insertion_mode"
        ] == row["insertion_runtime_mode"]
        assert protocol["route_contract"]["execution_settings"][
            "phase1_shortlist_size"
        ] == 24
        assert protocol["route_contract"]["execution_settings"][
            "phase2_shortlist_size"
        ] == 12
        assert protocol["route_contract"]["semantic_invariants"][
            "admission_cardinality"
        ] == 1
        pool = protocol["executable_pool"]
        expected = contract.GLOBAL_POOL_BY_NPH[row["nph"]]
        assert pool["count"] == expected["count"]
        assert pool["ordered_labels_sha256"] == (
            expected["ordered_labels_sha256"]
        )
        assert pool["ordered_pool_sha256"] == (
            contract.ORDERED_POOL_SHA256_BY_REGIME[
                row["regime_id"]
            ]
        )


def test_semantic_preflight_and_open_plateau_calibration() -> None:
    contract = _load_contract()
    smoke = _load(
        PACKAGE_DIR / "two_round_semantic_preflight_receipt.json"
    )
    contract.validate_smoke_receipt(smoke)
    assert len(smoke["observations"]) == 2
    assert {
        row["route_id"] for row in smoke["observations"]
    } == set(contract.ROUTE_IDS)
    assert all(
        row["controller_round_count"] == 2
        and row["global_pool_count"] == 6508
        and row["scientific_result"] is False
        and row["execution_evidence"] is False
        for row in smoke["observations"]
    )

    calibration = _load(
        PACKAGE_DIR / "plateau_open_domain_calibration_receipt.json"
    )
    contract.validate_calibration_receipt(calibration)
    domain = calibration["open_domain_receipt"]
    assert calibration["candidate_count"] == 6508
    assert calibration[
        "precollapse_candidate_position_pair_count"
    ] == 13016
    assert domain["domain_open"] is True
    assert domain["requested_positions"] == [0, 1]
    assert (
        domain["retained_representative_count"]
        + domain["collapsed_position_count"]
        == 13016
    )
    assert calibration["package_resources_demonstrated"] is False
    assert calibration["package_resource_status"] == (
        "provisional_not_demonstrated"
    )


def test_package_has_12_inert_jobs_and_neutral_queue_items() -> None:
    contract = _load_contract()
    authority = contract.validate_materialization_authority(
        REPO_ROOT
    )
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    contract.verify_self_digest(manifest, label="package manifest")
    assert manifest["status"] == "passed"
    assert manifest["direct_execution_count"] == 12
    assert manifest["insertion_policy_count"] == 2
    assert manifest["resource_status"] == (
        "provisional_not_demonstrated"
    )
    assert manifest["authority_overlay_present"] is False
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert not (PACKAGE_DIR / "authority").exists()
    assert len(list((PACKAGE_DIR / "jobs").glob("*.json"))) == 12
    for row in contract.direct_execution_rows():
        job = _load(
            PACKAGE_DIR
            / "jobs"
            / f"{row['execution_id']}.json"
        )
        assert job["protocol"] == authority["protocol_bindings"][
            row["execution_id"]
        ]
        assert job["resources"] == row["resources"]
        assert job["execution_authorized"] is False
        assert job["submission_authorized"] is False
        assert job["submitted"] is False

    queue = (PACKAGE_DIR / "queue.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(queue) == 12
    assert all(len(line.split("\t")) == 5 for line in queue)
    submit = (PACKAGE_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    assert (
        "queue execution_id,cpus,memory_mb,disk_mb,"
        "max_runtime_seconds from queue.tsv"
    ) in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "queue execution_id,request_" not in submit


def test_package_read_only_validator_passes() -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_DIR / "validate_package.py"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["direct_execution_count"] == 12
    assert payload["resource_status"] == (
        "provisional_not_demonstrated"
    )
    assert payload["execution_authorized"] is False
    assert payload["submission_authorized"] is False
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert not list(PACKAGE_DIR.rglob("__pycache__"))
