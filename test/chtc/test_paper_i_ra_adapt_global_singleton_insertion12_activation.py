from __future__ import annotations

from collections import Counter
import copy
import importlib.util
import json
import os
import re
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
    / "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc"
)
ACTIVATION_DIR = (
    REPAIR_ROOT
    / "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc_activation"
)
EXPECTED_PACKAGE_ID = (
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc"
)
EXPECTED_BATCH_NAME = (
    "paper-i-ra-adapt-global-singleton-insertion12-r50-20260730-v1"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "a7e8663e0b9daa3b7589652179e8e6ec6ebb7d4ad47925a294beed232b268940"
)
EXPECTED_PACKAGE_MANIFEST_FILE_SHA256 = (
    "b77b21f39e38a27b633b5c6b02e358a8c00cb8b02d0863d0f54922f3c4c7a838"
)
EXPECTED_EXECUTION_PLAN_SHA256 = (
    "3a2be2ffc22efc3896c80f8f00b65baeed1595353ad47c51a30f2bea7df0b85a"
)
EXPECTED_EXECUTION_PLAN_FILE_SHA256 = (
    "c72bcf2c30ebcd678ef5b57cd19a2c5eaff5a2e0d3f83dfe0745be956b48bc1c"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "2705bc4c424b9d9e4b116d2e3fe061359c3704ba2f504ac113e35d15c23411ac"
)
EXPECTED_CALIBRATION_SHA256 = (
    "98dec786b814a68ac7517325004d702e04123c048b55b9ccb0363100be94403b"
)
EXPECTED_CALIBRATION_FILE_SHA256 = (
    "1f5492655411dcf6e00090fbb4a41c147c3a3bbdca9bd621f16be7ba0c2cee20"
)
EXPECTED_ACTIVATION_MANIFEST_SHA256 = (
    "3f259919839abd9c070e635b049fc7a7923f718dc09a3eccddcf28ce1d379d86"
)
EXPECTED_QUEUE_VARIABLES = (
    "execution_id",
    "job_path",
    "job_file_sha256",
    "authorization_path",
    "authorization_file_sha256",
    "cpus",
    "memory_mb",
    "disk_mb",
    "max_runtime_seconds",
)
EXPECTED_ROUTES = {
    "ra_global_singleton_append_commutation_reduced",
    "ra_global_singleton_plateau_commutation",
}
EXPECTED_REGIMES = {
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = ACTIVATION_DIR / "activation_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_global_singleton_insertion12_activation_contract",
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


@pytest.fixture(scope="module")
def contract() -> ModuleType:
    return _load_contract()


@pytest.fixture(scope="module")
def activation_result(contract: ModuleType) -> dict[str, Any]:
    return contract.validate_activation(REPO_ROOT)


def _expanded_dry_run_text(
    executions: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for proc_id, execution in enumerate(executions):
        resources = execution["resources"]
        blocks.append(
            "\n".join(
                (
                    "[",
                    f"ProcId = {proc_id}",
                    f"RequestCpus = {resources['request_cpus']}",
                    f"RequestMemory = {resources['request_memory_mb']}",
                    (
                        "RequestDisk = "
                        f"{resources['request_disk_mb'] * 1024}"
                    ),
                    (
                        "MaxRuntime = "
                        f"{resources['max_runtime_seconds']}"
                    ),
                    f'JobBatchName = "{EXPECTED_BATCH_NAME}"',
                    "LeaveJobInQueue = true",
                    "]",
                )
            )
        )
    return "\n".join(blocks) + "\n"


def _expanded_condor_delta_dry_run_text(
    executions: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    baseline: dict[str, int | str] = {}
    for proc_id, execution in enumerate(executions):
        resources = execution["resources"]
        values: dict[str, int | str] = {
            "ProcId": proc_id,
            "RequestCpus": resources["request_cpus"],
            "RequestMemory": resources["request_memory_mb"],
            "RequestDisk": resources["request_disk_mb"] * 1024,
            "MaxRuntime": resources["max_runtime_seconds"],
            "JobBatchName": f'"{EXPECTED_BATCH_NAME}"',
            "LeaveJobInQueue": "true",
        }
        lines = [
            f"{name}={value}"
            for name, value in values.items()
            if name == "ProcId"
            or proc_id == 0
            or baseline.get(name) != value
        ]
        blocks.append("\n".join(lines))
        if proc_id == 0:
            baseline = dict(values)
    return "\n\n".join(blocks) + "\n"


def _factory_dry_run_text(*, cluster_id: int = 37) -> str:
    return "\n".join(
        (
            f"ClusterId = {cluster_id}",
            f'JobBatchName = "{EXPECTED_BATCH_NAME}"',
            "LeaveJobInQueue = true",
            'Requirements = split(TARGET.CondorVersion)[1] != ""',
            "",
        )
    )


def test_package_and_activation_bytes_are_exact(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    plan = _load(PACKAGE_DIR / "execution_plan.json")
    calibration = _load(
        PACKAGE_DIR / "plateau_open_domain_calibration_receipt.json"
    )
    activation = activation_result["manifest"]
    for digest in (
        EXPECTED_PACKAGE_MANIFEST_SHA256,
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256,
        EXPECTED_EXECUTION_PLAN_SHA256,
        EXPECTED_EXECUTION_PLAN_FILE_SHA256,
        EXPECTED_SOURCE_ARCHIVE_SHA256,
        EXPECTED_CALIBRATION_SHA256,
        EXPECTED_CALIBRATION_FILE_SHA256,
        EXPECTED_ACTIVATION_MANIFEST_SHA256,
    ):
        assert SHA256_RE.fullmatch(digest)
    assert manifest["package_id"] == EXPECTED_PACKAGE_ID
    assert manifest["sha256"] == EXPECTED_PACKAGE_MANIFEST_SHA256
    assert contract.sha256_file(
        PACKAGE_DIR / "package_manifest.json"
    ) == EXPECTED_PACKAGE_MANIFEST_FILE_SHA256
    assert plan["sha256"] == EXPECTED_EXECUTION_PLAN_SHA256
    assert contract.sha256_file(
        PACKAGE_DIR / "execution_plan.json"
    ) == EXPECTED_EXECUTION_PLAN_FILE_SHA256
    assert contract.sha256_file(
        PACKAGE_DIR / "source_locked.tar.gz"
    ) == EXPECTED_SOURCE_ARCHIVE_SHA256
    assert calibration["sha256"] == EXPECTED_CALIBRATION_SHA256
    assert contract.sha256_file(
        PACKAGE_DIR / "plateau_open_domain_calibration_receipt.json"
    ) == EXPECTED_CALIBRATION_FILE_SHA256
    assert manifest["direct_execution_count"] == 12
    assert len(manifest["jobs"]) == 12
    assert len(plan["execution_ids"]) == 12
    assert len(set(plan["execution_ids"])) == 12
    assert activation["sha256"] == EXPECTED_ACTIVATION_MANIFEST_SHA256


def test_activation_authorizes_exact_12_cell_comparison(
    activation_result: dict[str, Any],
) -> None:
    manifest = activation_result["manifest"]
    jobs = [
        _load(REPO_ROOT / execution["job"]["path"])
        for execution in manifest["executions"]
    ]

    assert Counter(job["route_id"] for job in jobs) == {
        route: 6 for route in EXPECTED_ROUTES
    }
    assert Counter(job["regime_id"] for job in jobs) == {
        regime: 2 for regime in EXPECTED_REGIMES
    }
    assert all(
        job["candidate_representation"] == "single_pauli_word_v1"
        and job["active_gradient_policy"]
        == "stationary_source_response_v1"
        and job["resource_weighting_scope"]
        == "all_phase_resource_weighting_v1"
        and job["phase1_cost_term"] == "enabled"
        for job in jobs
    )
    assert manifest["batch_name"] == EXPECTED_BATCH_NAME
    assert manifest["direct_execution_count"] == 12
    assert manifest["resource_status"] == (
        "provisional_not_demonstrated"
    )
    assert manifest["queue_variables"] == list(
        EXPECTED_QUEUE_VARIABLES
    )
    assert manifest["execution_authorized"] is True
    assert manifest["submission_authorized"] is True
    assert manifest["submission_state"] == "authorized_not_submitted"
    assert manifest["remote_stage"] is False
    assert manifest["condor_submit"] is False
    assert manifest["submitted"] is False
    assert manifest["paper_evidence_adopted"] is False
    assert "cluster_id" not in manifest
    assert len(activation_result["authorizations"]) == 12


def test_provisional_calibration_is_bound_and_valid(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    calibration = activation_result["calibration"]
    contract.validate_provisional_calibration_payload(calibration)
    assert calibration["package_resources_demonstrated"] is False
    assert calibration["package_resource_status"] == (
        "provisional_not_demonstrated"
    )
    assert calibration["candidate_count"] == 6508
    assert calibration["requested_positions"] == [0, 1]

    tampered = copy.deepcopy(calibration)
    tampered["package_resource_status"] = "missing"
    tampered.pop("sha256")
    tampered = contract.digested(tampered)
    with pytest.raises(
        contract.ActivationContractError,
        match="provisional calibration",
    ):
        contract.validate_provisional_calibration_payload(tampered)


def test_external_authorizations_bind_jobs_and_provisional_resources(
    activation_result: dict[str, Any],
) -> None:
    manifest = activation_result["manifest"]
    authorization_paths = sorted(
        (ACTIVATION_DIR / "authorizations").glob("*.json")
    )
    assert len(authorization_paths) == 12
    assert not list(
        (ACTIVATION_DIR / "authorizations").glob("**/*.tmp")
    )
    for execution, authorization in zip(
        manifest["executions"],
        activation_result["authorizations"],
        strict=True,
    ):
        job = _load(REPO_ROOT / execution["job"]["path"])
        assert authorization["execution_id"] == job["execution_id"]
        assert authorization["job_sha256"] == job["sha256"]
        assert authorization["job_file_sha256"] == (
            execution["job"]["sha256"]
        )
        assert authorization["package_manifest_sha256"] == (
            EXPECTED_PACKAGE_MANIFEST_SHA256
        )
        assert authorization["execution_plan_sha256"] == (
            EXPECTED_EXECUTION_PLAN_SHA256
        )
        assert authorization["source_archive_sha256"] == (
            EXPECTED_SOURCE_ARCHIVE_SHA256
        )
        assert authorization["open_plateau_calibration_sha256"] == (
            EXPECTED_CALIBRATION_SHA256
        )
        assert authorization["resource_status"] == (
            "provisional_not_demonstrated"
        )
        assert execution["resources"] == job["resources"]
        assert authorization["execution_authorized"] is True
        assert authorization["submission_authorized"] is True
        assert authorization["submission_state"] == (
            "authorized_not_submitted"
        )
        assert authorization["remote_stage"] is False
        assert authorization["condor_submit"] is False
        assert authorization["submitted"] is False


def test_authorization_tamper_is_rejected(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    manifest = activation_result["manifest"]
    execution = manifest["executions"][0]
    authorization = copy.deepcopy(
        activation_result["authorizations"][0]
    )
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


def test_submit_uses_only_neutral_item_variables(
    contract: ModuleType,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    contract.validate_submit_text(submit)
    assert f'+JobBatchName = "{EXPECTED_BATCH_NAME}"' in submit
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "request_cpus = $(cpus)" in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "+MaxRuntime = $(max_runtime_seconds)" in submit
    assert "max_materialize = 1" in submit
    assert "leave_in_queue = True" in submit
    assert contract.MAX_MATERIALIZE == 1
    assert contract.LEAVE_IN_QUEUE is True
    assert "$(request_" not in submit.lower()
    queue_line = next(
        line.strip()
        for line in submit.splitlines()
        if line.strip().lower().startswith("queue ")
    )
    variables = tuple(
        value.strip()
        for value in queue_line[6:].split(" from ", 1)[0].split(",")
    )
    assert variables == EXPECTED_QUEUE_VARIABLES
    assert not any(
        value.lower().startswith("request_") for value in variables
    )
    for forbidden in (
        "Requestmemory_mb",
        "Requestdisk_mb",
        "TARGET.memory_mb",
        "TARGET.disk_mb",
    ):
        assert forbidden.lower() not in submit.lower()


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            "authorization_file_sha256,cpus,memory_mb,disk_mb,"
            "max_runtime_seconds",
            "authorization_file_sha256,request_cpus,memory_mb,disk_mb,"
            "max_runtime_seconds",
        ),
        ("request_cpus = $(cpus)", "request_cpus = $(request_cpus)"),
    ),
)
def test_submit_rejects_request_prefixed_item_variables(
    contract: ModuleType,
    before: str,
    after: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    assert before in submit
    with pytest.raises(contract.ActivationContractError):
        contract.validate_submit_text(submit.replace(before, after, 1))


@pytest.mark.parametrize(
    ("before", "after"),
    (
        ("max_materialize = 1", "max_materialize = 2"),
        ("leave_in_queue = True", "leave_in_queue = False"),
    ),
)
def test_submit_rejects_factory_policy_drift(
    contract: ModuleType,
    before: str,
    after: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    assert before in submit
    with pytest.raises(contract.ActivationContractError):
        contract.validate_submit_text(submit.replace(before, after, 1))


@pytest.mark.parametrize(
    "appended",
    (
        "max_materialize = 1",
        "max_materialize = 2",
        "leave_in_queue = True",
        "leave_in_queue = False",
        "max_idle = 1",
        "+JobMaterializeLimit = 1",
        "JobMaterializeLimit = 1",
        "MY.JobMaterializeLimit = 1",
        "JobMaterializeMaxIdle = 1",
    ),
)
def test_submit_rejects_appended_competing_factory_policy(
    contract: ModuleType,
    appended: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    with pytest.raises(contract.ActivationContractError):
        contract.validate_submit_text(f"{submit.rstrip()}\n{appended}\n")


def test_expanded_dry_run_projection_removes_only_factory_limit(
    contract: ModuleType,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    projected = contract.expanded_dry_run_submit_text(submit)

    assert projected == submit.replace("max_materialize = 1\n", "", 1)
    assert "max_materialize" not in projected.lower()
    assert "leave_in_queue = True" in projected
    assert "max_materialize = 1" in submit


def _post_submit_factory_expectations() -> dict[str, Any]:
    return {
        "required": True,
        "observed_in_pre_submit_dry_run": False,
        "JobMaterializeLimit": 1,
        "TotalSubmitProcs": 12,
    }


def test_remote_expanded_dry_run_accepts_exactly_12_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _expanded_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    receipt = contract.validate_remote_expanded_dry_run_text(
        text,
        executions=activation_result["manifest"]["executions"],
    )
    assert receipt["status"] == "passed"
    assert receipt["kind"] == contract.EXPANDED_DRY_RUN_KIND
    assert receipt["ad_count"] == 12
    assert receipt["proc_ids"] == list(range(12))
    assert receipt["leave_in_queue"] is True
    assert receipt["post_submit_factory_expectations"] == (
        _post_submit_factory_expectations()
    )
    assert "factory_max_materialize" not in receipt
    assert len(receipt["resources"]) == 12


def test_remote_expanded_dry_run_accepts_condor_25_delta_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _expanded_condor_delta_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    receipt = contract.validate_remote_expanded_dry_run_text(
        text,
        executions=activation_result["manifest"]["executions"],
    )

    assert receipt["status"] == "passed"
    assert receipt["ad_count"] == 12
    assert receipt["proc_ids"] == list(range(12))
    assert len(receipt["resources"]) == 12


def test_remote_expanded_dry_run_rejects_omitted_delta_override(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _expanded_condor_delta_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    assert "ProcId=6\nRequestMemory=90112" in text

    with pytest.raises(
        contract.ActivationContractError,
        match="resource drift",
    ):
        contract.validate_remote_expanded_dry_run_text(
            text.replace(
                "ProcId=6\nRequestMemory=90112",
                "ProcId=6",
                1,
            ),
            executions=activation_result["manifest"]["executions"],
        )


def test_remote_expanded_dry_run_rejects_retention_drift(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _expanded_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    with pytest.raises(
        contract.ActivationContractError,
        match="completion-retention",
    ):
        contract.validate_remote_expanded_dry_run_text(
            text.replace(
                "LeaveJobInQueue = true",
                "LeaveJobInQueue = false",
                1,
            ),
            executions=activation_result["manifest"]["executions"],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            "ProcId = 11",
            "ProcId = 10",
            1,
        ),
        lambda value: value.replace(
            "RequestDisk = ",
            "RequestDisk = 1",
            1,
        ),
        lambda value: value.replace(
            "RequestMemory = ",
            "Requestmemory_mb = ",
            1,
        ),
    ),
)
def test_remote_expanded_dry_run_rejects_malformed_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
    mutation: Any,
) -> None:
    text = _expanded_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    with pytest.raises(contract.ActivationContractError):
        contract.validate_remote_expanded_dry_run_text(
            mutation(text),
            executions=activation_result["manifest"]["executions"],
        )


def test_remote_factory_dry_run_accepts_one_cluster_ad(
    contract: ModuleType,
) -> None:
    receipt = contract.validate_remote_factory_dry_run_text(
        _factory_dry_run_text(cluster_id=9042)
    )

    assert receipt["status"] == "passed"
    assert receipt["kind"] == contract.FACTORY_DRY_RUN_KIND
    assert receipt["cluster_ad_count"] == 1
    assert receipt["cluster_id"] == 9042
    assert receipt["batch_name"] == EXPECTED_BATCH_NAME
    assert receipt["leave_in_queue"] is True
    assert receipt["post_submit_factory_expectations"] == (
        _post_submit_factory_expectations()
    )
    assert "JobMaterializeLimit" not in receipt
    assert "TotalSubmitProcs" not in receipt


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: f"{value.rstrip()}\nProcId = 0\n",
        lambda value: value.replace(
            EXPECTED_BATCH_NAME,
            "wrong-batch",
            1,
        ),
        lambda value: value.replace(
            "LeaveJobInQueue = true",
            "LeaveJobInQueue = false",
            1,
        ),
        lambda value: value.replace("ClusterId = 37\n", "", 1),
        lambda value: f"{value}\n{value}",
    ),
)
def test_remote_factory_dry_run_rejects_invalid_cluster_ad(
    contract: ModuleType,
    mutation: Any,
) -> None:
    with pytest.raises(contract.ActivationContractError):
        contract.validate_remote_factory_dry_run_text(
            mutation(_factory_dry_run_text())
        )


def test_activation_read_only_validator_passes(
    activation_result: dict[str, Any],
    tmp_path: Path,
) -> None:
    expanded_path = tmp_path / "singleton12.expanded.dry-run.classads"
    expanded_path.write_text(
        _expanded_dry_run_text(
            activation_result["manifest"]["executions"]
        ),
        encoding="utf-8",
    )
    factory_path = tmp_path / "singleton12.factory.dry-run.classad"
    factory_path.write_text(
        _factory_dry_run_text(cluster_id=9042),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ACTIVATION_DIR / "validate_activation.py"),
            "--remote-expanded-dry-run",
            str(expanded_path),
            "--remote-factory-dry-run",
            str(factory_path),
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
    assert payload["submission_state"] == "authorized_not_submitted"
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert payload["submitted"] is False
    expanded = payload["remote_expanded_dry_run_validation"]
    factory = payload["remote_factory_dry_run_validation"]
    assert expanded["status"] == "passed"
    assert expanded["kind"] == "expanded_nonfactory_projection_v1"
    assert expanded["ad_count"] == 12
    assert expanded["proc_ids"] == list(range(12))
    assert expanded["post_submit_factory_expectations"] == (
        _post_submit_factory_expectations()
    )
    assert factory["status"] == "passed"
    assert factory["kind"] == "factory_cluster_ad_v1"
    assert factory["cluster_ad_count"] == 1
    assert factory["cluster_id"] == 9042
    assert factory["post_submit_factory_expectations"] == (
        _post_submit_factory_expectations()
    )
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))


@pytest.mark.parametrize(
    "argument",
    ("--remote-expanded-dry-run", "--remote-factory-dry-run"),
)
def test_activation_cli_requires_dual_dry_runs(
    tmp_path: Path,
    argument: str,
) -> None:
    dry_run_path = tmp_path / "one.dry-run.classad"
    dry_run_path.write_text(_factory_dry_run_text(), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ACTIVATION_DIR / "validate_activation.py"),
            argument,
            str(dry_run_path),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "are required together" in completed.stderr
