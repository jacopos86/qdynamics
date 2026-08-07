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
    REPAIR_ROOT / "stationary_ra_always12_r50_20260729_v2_chtc"
)
ACTIVATION_DIR = (
    REPAIR_ROOT
    / "stationary_ra_always12_r50_20260729_v2_chtc_activation"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "467be866ac8abd01b109aefea69112aacb1b658da37c66eac92a4976d387fe9f"
)
EXPECTED_PACKAGE_MANIFEST_FILE_SHA256 = (
    "f1c0b0abe107d6a35882ef11f4a649d20c9e08c16719b30e59226ff29e6813fb"
)
EXPECTED_EXECUTION_PLAN_SHA256 = (
    "df3f215c66901fbac868fd7253e6d6f2a5edd97d7d2e43fe5aa9deb41dc5d45a"
)
EXPECTED_EXECUTION_PLAN_FILE_SHA256 = (
    "a9679495b5c664877004cad7478b216d5085251bf5a153c24c9560bde6b2963e"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "1407947832291ab15ad91b0455058a6de689dac42cd1cb5282a76eeafbbc409d"
)
EXPECTED_ACTIVATION_MANIFEST_SHA256 = (
    "b6f00caf09750722bdab675239b58e5f0da37839e5704c4f46902f332c6dc04f"
)
EXPECTED_BATCH_NAME = (
    "paper-i-ra-adapt-stationary-ra-always12-r50-20260729-v2"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = ACTIVATION_DIR / "activation_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_adapt_always12_v2_activation_contract", path
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


def test_sealed_v2_bytes_remain_exact() -> None:
    contract = _load_contract()
    inventory = contract.sealed_package_inventory(PACKAGE_DIR)

    assert len(inventory) == 25
    assert contract.sha256_file(PACKAGE_DIR / "package_manifest.json") == (
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256
    )
    assert _load(PACKAGE_DIR / "package_manifest.json")["sha256"] == (
        EXPECTED_PACKAGE_MANIFEST_SHA256
    )
    assert contract.sha256_file(PACKAGE_DIR / "execution_plan.json") == (
        EXPECTED_EXECUTION_PLAN_FILE_SHA256
    )
    assert _load(PACKAGE_DIR / "execution_plan.json")["sha256"] == (
        EXPECTED_EXECUTION_PLAN_SHA256
    )
    assert contract.sha256_file(PACKAGE_DIR / "source_locked.tar.gz") == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )


def test_activation_authorizes_exact_batch_without_submission_claim() -> None:
    contract = _load_contract()
    result = contract.validate_activation(REPO_ROOT)
    manifest = result["manifest"]

    assert manifest["sha256"] == EXPECTED_ACTIVATION_MANIFEST_SHA256
    assert manifest["batch_name"] == EXPECTED_BATCH_NAME
    assert manifest["direct_execution_count"] == 12
    assert manifest["execution_authorized"] is True
    assert manifest["submission_authorized"] is True
    assert manifest["submission_state"] == "authorized_not_submitted"
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False
    assert manifest["submitted"] is False
    assert "cluster_id" not in manifest
    assert len(result["authorizations"]) == 12


def test_each_authorization_binds_exact_sealed_job() -> None:
    contract = _load_contract()
    result = contract.validate_activation(REPO_ROOT)
    manifest = result["manifest"]

    for row, authorization in zip(
        manifest["executions"], result["authorizations"], strict=True
    ):
        job = _load(REPO_ROOT / row["job"]["path"])
        assert authorization["execution_id"] == job["execution_id"]
        assert authorization["job_sha256"] == job["sha256"]
        assert authorization["package_manifest_sha256"] == (
            EXPECTED_PACKAGE_MANIFEST_SHA256
        )
        assert authorization["execution_plan_sha256"] == (
            EXPECTED_EXECUTION_PLAN_SHA256
        )
        assert authorization["source_archive_sha256"] == (
            EXPECTED_SOURCE_ARCHIVE_SHA256
        )
        assert authorization["execution_authorized"] is True
        assert authorization["submission_authorized"] is True
        assert authorization["submission_state"] == (
            "authorized_not_submitted"
        )
        assert authorization["remote_stage"] is False
        assert authorization["condor_submit"] is False
        assert authorization["submitted"] is False


def test_authorization_tamper_is_rejected() -> None:
    contract = _load_contract()
    result = contract.validate_activation(REPO_ROOT)
    manifest = result["manifest"]
    execution = manifest["executions"][0]
    authorization = copy.deepcopy(result["authorizations"][0])
    authorization["job_sha256"] = "0" * 64
    authorization.pop("sha256")
    authorization = contract.digested(authorization)

    with pytest.raises(
        contract.ActivationContractError,
        match="[Aa]uthorization binding drifted",
    ):
        contract.validate_authorization_payload(
            authorization,
            execution=execution,
            manifest=manifest,
        )


def test_submit_and_wrapper_close_operational_bindings() -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    wrapper = (ACTIVATION_DIR / "execute_authorized_job.sh").read_text(
        encoding="utf-8"
    )

    assert f'+JobBatchName = "{EXPECTED_BATCH_NAME}"' in submit
    assert "requirements = TARGET.HasSIF" in submit
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "preserve_relative_paths = True" in submit
    assert "stream_output = False" in submit
    assert "stream_error = False" in submit
    assert "$(authorization_path)" in submit
    assert "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz" in submit
    assert "transfer_output_remaps" in submit
    assert "queue execution_id," in submit
    assert "command -v apptainer" in wrapper
    assert "command -v singularity" in wrapper
    assert '"$runtime_bin" exec' in wrapper
    assert '--pwd "$worker_abs"' in wrapper
    assert "expected_image_sha256" in wrapper
    assert "expected_source_sha256" in wrapper
    assert "_CONDOR_JOB_AD" in wrapper


def test_activation_read_only_validator_passes() -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [sys.executable, str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "passed"
    assert payload["direct_execution_count"] == 12
    assert payload["execution_authorized"] is True
    assert payload["submission_authorized"] is True
    assert payload["submission_state"] == "authorized_not_submitted"
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert payload["submitted"] is False
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))
