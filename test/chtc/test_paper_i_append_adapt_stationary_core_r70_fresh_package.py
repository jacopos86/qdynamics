from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = (
    REPAIR_ROOT
    / "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc"
)
VISIBLE_R50_PACKAGE_DIR = (
    REPAIR_ROOT / "stationary_core_full48_r50_20260728_v6_chtc"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "eea38b59e60d727281dc3bdaf6d2efa7880f3f49375ce49e61134fbb35a566ea"
)
EXPECTED_EXECUTION_PLAN_SHA256 = (
    "8289b35f84220ac5704e1eff4349f0e243c06a53136766c6a386e63657f34dc8"
)
EXPECTED_HORIZON_DELTA_AUDIT_SHA256 = (
    "7f6bcf0cc8e12f69e77fdf10260f3c854b2afee65d6f66e268432355ad15f74e"
)
EXPECTED_ANCHOR_EVIDENCE_SHA256 = (
    "5241f3ad71799f9ca139a0921ed4ae626c6ae396f77625626acf1f0ec98a9cef"
)
EXPECTED_VISIBLE_SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_contract() -> ModuleType:
    path = PACKAGE_DIR / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_append_adapt_stationary_core_r70_fresh_contract",
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


def test_read_only_full_archive_validator_passes_for_inert_package() -> None:
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
            "--full-archive-scan",
            "--full-anchor-scan",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "activation_required_before_submission": True,
        "anchor_evidence_sha256": EXPECTED_ANCHOR_EVIDENCE_SHA256,
        "anchor_full_worker_attempt_passed": False,
        "anchor_independent_execution": True,
        "anchor_reproduces_source": True,
        "anchor_reproduction_scope": "scientific_payload_only",
        "anchor_worker_exit_status": 2,
        "condor_submit": False,
        "direct_execution_count": 12,
        "execution_authorized": False,
        "execution_plan_sha256": EXPECTED_EXECUTION_PLAN_SHA256,
        "fresh_start": True,
        "horizon_delta_audit_sha256": (
            EXPECTED_HORIZON_DELTA_AUDIT_SHA256
        ),
        "package_id": (
            "paper_i_append_adapt_stationary_core12_r70_fresh_"
            "20260731_v1_chtc"
        ),
        "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA256,
        "remote_stage": False,
        "source_archive_sha256": EXPECTED_VISIBLE_SOURCE_ARCHIVE_SHA256,
        "source_full_worker_attempt_passed": True,
        "source_horizon": 50,
        "source_worker_exit_status": 0,
        "status": "passed",
        "submission_authorized": False,
        "submission_state": "not_submitted",
        "target_horizon": 70,
    }
    assert not (PACKAGE_DIR / "submit.sub").exists()
    assert not (PACKAGE_DIR / "execute_source_locked_job.sh").exists()
    assert not (PACKAGE_DIR / "authority").exists()
    assert not list(PACKAGE_DIR.rglob("__pycache__"))


def test_twelve_rows_are_fresh_horizon_only_extensions_of_visible_v6() -> None:
    contract = _load_contract()
    package = contract.validate_package(full_archive_scan=False)
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    plan = _load(PACKAGE_DIR / "execution_plan.json")
    audit = _load(PACKAGE_DIR / "horizon_delta_audit.json")
    anchor_evidence = _load(PACKAGE_DIR / "anchor_evidence.json")
    source_authority = _load(PACKAGE_DIR / "source_authority.json")

    assert package["package_manifest_sha256"] == (
        EXPECTED_PACKAGE_MANIFEST_SHA256
    )
    assert manifest["source_archive"]["sha256"] == (
        EXPECTED_VISIBLE_SOURCE_ARCHIVE_SHA256
    )
    assert plan["source_horizon"] == 50
    assert plan["target_horizon"] == 70
    assert plan["fresh_start"] is True
    assert plan["resume_claimed"] is False
    assert source_authority["source_package_id"] == (
        "paper_i_ra_adapt_stationary_core_full48_r50_20260728_v6_chtc"
    )
    assert source_authority["source_archive_sha256"] == (
        EXPECTED_VISIBLE_SOURCE_ARCHIVE_SHA256
    )
    assert anchor_evidence["sha256"] == EXPECTED_ANCHOR_EVIDENCE_SHA256
    assert anchor_evidence["anchor_execution"]["archive"] == {
        "path": (
            "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v5_"
            "9392023_20260729/core__weak_weak__nph3__append_singleton__"
            "cluster_9392023__proc_4.tar.gz"
        ),
        "sha256": (
            "7b0183478c04e5874af83f5cc3fde66d6105708993b58814d7c4c154576b24d8"
        ),
        "size_bytes": 177_074_477,
    }
    assert anchor_evidence["source_execution"]["archive"] == {
        "path": (
            "raw_outputs/chtc_fetch_paper_i_ra_adapt_stationary_core_v6_"
            "9392337_20260729/core__weak_weak__nph3__append_singleton__"
            "cluster_9392337__proc_4.tar.gz"
        ),
        "sha256": (
            "3f8d07ea935f156de03490c94f4e85f794f0809511e56f3f37e56b249d825490"
        ),
        "size_bytes": 177_079_366,
    }
    assert anchor_evidence["anchor_execution"]["worker_exit_status"] == 2
    assert (
        anchor_evidence["anchor_execution"]["full_worker_attempt_passed"]
        is False
    )
    assert anchor_evidence["anchor_execution"]["worker_receipt"] is None
    assert anchor_evidence["source_execution"]["worker_exit_status"] == 0
    assert (
        anchor_evidence["source_execution"]["full_worker_attempt_passed"]
        is True
    )
    assert (
        anchor_evidence["source_execution"]["worker_receipt"]["sha256"]
        == "a80aa2e6a67ece5ff5250d173bd9e806df130ca6c3dae20d8da3276a85a4be56"
    )
    comparison = anchor_evidence["comparison"]
    assert comparison["scientific_payload_reproduces_source"] is True
    assert comparison["full_worker_attempt_reproduces_source"] is False
    assert comparison["result_bytes_match"] is True
    assert comparison["summary_bytes_match"] is True
    assert comparison["checkpoint_bytes_match"] is True
    assert comparison["operator_sequence_match"] is True
    assert comparison["generator_sequence_match"] is True
    assert comparison["metric_match"] is True
    assert comparison["metric_abs_diff"] == 0.0
    assert comparison["stopping_condition_match"] is True
    assert comparison["stop_reason"] == "maximum_controller_rounds"
    control_delta = comparison["run_cell_control_plane_delta"]
    assert control_delta["changed_top_level_functions"] == [
        "_compiled_resource_projection",
        "_run_g11_bounded_diagnostic",
        "_validate_g6_ra_round",
    ]
    assert control_delta["scientific_primary_invoke_changed"] is False
    assert comparison["module_non_function_ast_sha256_match"] is True
    assert comparison["module_non_function_ast_sha256"] == (
        "6e7b7bc631b91efb0916d7f5472e4fcbe44f08229328f31085f480cacb392ed3"
    )
    assert control_delta["failure_stage_bound"] == (
        "after_primary_scientific_payload_before_worker_receipt"
    )
    assert control_delta["failure_cause_asserted"] is False
    assert audit["anchor"]["reproduction_scope"] == (
        "scientific_payload_only"
    )
    assert audit["anchor"]["anchor_full_worker_attempt_passed"] is False

    expected_matrix = {
        (regime_id, nph, route_id)
        for regime_id, nph in contract.REGIME_CUTOFF_PAIRS
        for route_id in contract.ROUTE_IDS
    }
    rows = audit["planned_rows"]
    assert len(rows) == 12
    assert {
        (row["regime_id"], row["nph"], row["route_id"])
        for row in rows
    } == expected_matrix
    source_rows = {
        row["execution_id"]: row
        for row in source_authority["source_rows"]
    }
    planned_rows = {
        row["execution_id"]: row for row in plan["direct_executions"]
    }
    for row in rows:
        execution_id = row["execution_id"]
        source_row = source_rows[execution_id]
        job = _load(PACKAGE_DIR / f"jobs/{execution_id}.json")
        source_job_path = (
            VISIBLE_R50_PACKAGE_DIR / source_row["source_job"]["path"]
        )
        source_job = _load(source_job_path)

        assert source_row["source_protocol"]["path"].startswith(
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "bundles/materializations/ra_adapt_stationary_late_core_v10/"
        )
        assert contract.sha256_file(source_job_path) == (
            source_row["source_job"]["sha256"]
        )
        assert source_job_path.stat().st_size == (
            source_row["source_job"]["size_bytes"]
        )
        assert source_job["sha256"] == (
            source_row["source_job"]["canonical_sha256"]
        )
        assert job["resources"] == source_job["resources"]
        assert job["source_protocol"] == source_job["protocol"]
        assert job["fresh_start_contract"] == {
            "controller_round_origin": 0,
            "kind": "fresh_start",
            "resume_claimed": False,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
        }
        assert job["horizon"] == {"source": 50, "target": 70}
        assert planned_rows[execution_id]["job_spec_sha256"] == (
            job["sha256"]
        )
        assert row["changed_scalar_paths"] == list(
            contract.ALLOWED_PROTOCOL_DELTA_PATHS
        )
        assert row["normalized_non_horizon_settings_match"] is True
        assert row["fields_added_by_current_defaults"] == []
        assert row["unresolved_source_fields"] == []
        assert row["source_checkpoint_consumed"] is False
        assert row["source_result_consumed"] is False
        assert row["resume_claimed"] is False


def test_delta_contract_rejects_a_non_horizon_scientific_tamper() -> None:
    contract = _load_contract()
    audit = _load(PACKAGE_DIR / "horizon_delta_audit.json")
    source_authority = _load(PACKAGE_DIR / "source_authority.json")
    row = copy.deepcopy(audit["planned_rows"][0])
    source_row = next(
        item
        for item in source_authority["source_rows"]
        if item["execution_id"] == row["execution_id"]
    )
    row["candidate_representation"] = "single_pauli_word_v1"
    with pytest.raises(
        contract.PackageContractError,
        match="Horizon-only delta row drifted",
    ):
        contract._validate_delta_row(
            row,
            execution_id=row["execution_id"],
            source_authority_row=source_row,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("anchor_execution", "worker_exit_status"), 0),
        (
            ("anchor_execution", "result", "sha256"),
            "0" * 64,
        ),
        (("comparison", "operator_sequence_sha256"), "0" * 64),
        (("comparison", "stop_reason"), "rewritten_stop"),
        (("comparison", "anchor_metric"), 0.0),
        (
            (
                "comparison",
                "run_cell_control_plane_delta",
                "classification",
            ),
            "rewritten_control_plane_classification",
        ),
        (
            (
                "comparison",
                "run_cell_control_plane_delta",
                "changed_top_level_functions",
            ),
            [],
        ),
        (
            (
                "comparison",
                "run_cell_control_plane_delta",
                "scientific_protocol_or_result_changed",
            ),
            True,
        ),
        (
            ("comparison", "module_non_function_ast_sha256"),
            "0" * 64,
        ),
    ),
)
def test_anchor_contract_rejects_rewritten_material_claim(
    path: tuple[str, ...],
    value: object,
) -> None:
    contract = _load_contract()
    evidence = _load(PACKAGE_DIR / "anchor_evidence.json")
    target: dict[str, Any] = evidence
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    evidence.pop("sha256")
    tampered = contract.digested(evidence)
    with pytest.raises(
        contract.PackageContractError,
        match="Independent anchor evidence drifted",
    ):
        contract.validate_anchor_evidence(
            tampered,
            full_attempt_scan=False,
        )
