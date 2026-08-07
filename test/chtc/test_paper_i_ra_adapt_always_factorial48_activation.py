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
    REPAIR_ROOT / "ra_always_factorial48_r50_20260730_v1_chtc"
)
ACTIVATION_DIR = (
    REPAIR_ROOT
    / "ra_always_factorial48_r50_20260730_v1_chtc_activation"
)
EXPECTED_PACKAGE_ID = (
    "paper_i_ra_adapt_always_factorial48_r50_20260730_v1_chtc"
)
EXPECTED_BATCH_NAME = (
    "paper-i-ra-adapt-always-factorial48-r50-20260730-v1"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "9f66b50958001a359229ed2d70c90465f716f239c327a81c987d7fb6b7581092"
)
EXPECTED_PACKAGE_MANIFEST_FILE_SHA256 = (
    "ea2170a68521d01cb2ee865807b635b37365a70c152391d234fe5d48e793253c"
)
EXPECTED_EXECUTION_PLAN_SHA256 = (
    "6c3fa999fb3ed59c8c9d3e07cb51eb73692eaf9a0bc1e01c78866117eae09120"
)
EXPECTED_EXECUTION_PLAN_FILE_SHA256 = (
    "68644ee7e7482d922839726242e3e6b250970cd4ccdbfbbd3dbee6c7d4304a5a"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "efae5d26981bab62cfb2b6dbf077effb4f19a7da6f636dde2b16fbf7acde76b6"
)
EXPECTED_ACTIVATION_MANIFEST_SHA256 = (
    "7c4168d7138a52052115cabe6209cc3f2beb3d20cb0ef6fefc11b211d66c8936"
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
EXPECTED_FACTORIAL_ARMS = {
    (
        "stationary_source_response_v1",
        "late_resource_weighting_v1",
        "disabled_for_phase1_only",
    ),
    (
        "measured_residual_response_v1",
        "late_resource_weighting_v1",
        "disabled_for_phase1_only",
    ),
    (
        "stationary_source_response_v1",
        "all_phase_resource_weighting_v1",
        "enabled",
    ),
    (
        "measured_residual_response_v1",
        "all_phase_resource_weighting_v1",
        "enabled",
    ),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_contract() -> ModuleType:
    path = ACTIVATION_DIR / "activation_contract.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_always_factorial48_activation_contract", path
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


def _dry_run_text(executions: list[dict[str, Any]]) -> str:
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


def _condor_delta_dry_run_text(
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


def _factory_dry_run_text() -> str:
    return "\n".join(
        (
            "ClusterId=1",
            f'JobBatchName="{EXPECTED_BATCH_NAME}"',
            "LeaveJobInQueue=true",
            'Requirements=split(TARGET.CondorVersion)[1] != ""',
            "",
        )
    )


def test_factorial_package_bytes_and_activation_are_exact(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    manifest = _load(PACKAGE_DIR / "package_manifest.json")
    plan = _load(PACKAGE_DIR / "execution_plan.json")
    activation = activation_result["manifest"]

    for digest in (
        EXPECTED_PACKAGE_MANIFEST_SHA256,
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256,
        EXPECTED_EXECUTION_PLAN_SHA256,
        EXPECTED_EXECUTION_PLAN_FILE_SHA256,
        EXPECTED_SOURCE_ARCHIVE_SHA256,
        EXPECTED_ACTIVATION_MANIFEST_SHA256,
    ):
        assert SHA256_RE.fullmatch(digest)
    assert manifest["package_id"] == EXPECTED_PACKAGE_ID
    assert manifest["sha256"] == EXPECTED_PACKAGE_MANIFEST_SHA256
    assert contract.sha256_file(PACKAGE_DIR / "package_manifest.json") == (
        EXPECTED_PACKAGE_MANIFEST_FILE_SHA256
    )
    assert plan["sha256"] == EXPECTED_EXECUTION_PLAN_SHA256
    assert contract.sha256_file(PACKAGE_DIR / "execution_plan.json") == (
        EXPECTED_EXECUTION_PLAN_FILE_SHA256
    )
    assert contract.sha256_file(PACKAGE_DIR / "source_locked.tar.gz") == (
        EXPECTED_SOURCE_ARCHIVE_SHA256
    )
    assert manifest["direct_execution_count"] == 48
    assert len(manifest["jobs"]) == 48
    assert len(plan["execution_ids"]) == 48
    assert len(set(plan["execution_ids"])) == 48
    assert plan["execution_ids"][0] == (
        "core__weak_weak__nph3__ra_macro_always__"
        "gradient_stationary__phase1_cost_off"
    )
    assert plan["execution_ids"][-1] == (
        "core__strong_strong_u8__nph7__ra_singleton_always__"
        "gradient_measured__phase1_cost_on"
    )
    assert len(contract.package_inventory(PACKAGE_DIR)) == 61
    assert activation["sha256"] == EXPECTED_ACTIVATION_MANIFEST_SHA256


def test_activation_authorizes_the_full_cartesian_product(
    activation_result: dict[str, Any],
) -> None:
    manifest = activation_result["manifest"]
    jobs = [
        _load(REPO_ROOT / execution["job"]["path"])
        for execution in manifest["executions"]
    ]
    arm_counts = Counter(
        (
            job["active_gradient_policy"],
            job["resource_weighting_scope"],
            job["phase1_cost_term"],
        )
        for job in jobs
    )

    assert set(arm_counts) == EXPECTED_FACTORIAL_ARMS
    assert set(arm_counts.values()) == {12}
    assert Counter(job["route_id"] for job in jobs) == {
        "ra_macro_always": 24,
        "ra_singleton_always": 24,
    }
    assert Counter(job["regime_id"] for job in jobs) == {
        "weak_weak": 8,
        "intermediate_weak": 8,
        "strong_weak_u8": 8,
        "weak_strong": 8,
        "intermediate_strong": 8,
        "strong_strong_u8": 8,
    }
    assert all(
        job["source_lock_id"]
        == (
            f"{job['regime_id']}__nph{job['nph']}__"
            f"{job['route_id']}"
        )
        for job in jobs
    )
    assert manifest["batch_name"] == EXPECTED_BATCH_NAME
    assert manifest["direct_execution_count"] == 48
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
    assert len(activation_result["authorizations"]) == 48


def test_all_external_authorizations_bind_exact_jobs_and_resources(
    activation_result: dict[str, Any],
) -> None:
    manifest = activation_result["manifest"]
    authorization_paths = sorted(
        (ACTIVATION_DIR / "authorizations").glob("*.json")
    )

    assert len(authorization_paths) == 48
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
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")

    contract.validate_submit_text(submit)
    assert f'+JobBatchName = "{EXPECTED_BATCH_NAME}"' in submit
    assert "when_to_transfer_output = ON_EXIT" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "request_cpus = $(cpus)" in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "+MaxRuntime = $(max_runtime_seconds)" in submit
    assert "max_materialize = 4" in submit
    assert "leave_in_queue = True" in submit
    assert contract.MAX_MATERIALIZE == 4
    assert contract.LEAVE_IN_QUEUE is True
    assert "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz" in submit
    assert (
        "$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"
        in submit
    )
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


def test_wrapper_binds_attempt_identity_source_and_sif(
    contract: ModuleType,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    wrapper = (ACTIVATION_DIR / "execute_authorized_job.sh").read_text(
        encoding="utf-8"
    )

    assert contract.REMOTE_IMAGE_PATH in submit
    assert contract.REMOTE_IMAGE_SHA256 in submit
    assert EXPECTED_SOURCE_ARCHIVE_SHA256 in submit
    assert "command -v apptainer" in wrapper
    assert "command -v singularity" in wrapper
    assert "expected_image_sha256" in wrapper
    assert "expected_source_sha256" in wrapper
    assert "expected_authorization_file_sha256" in wrapper
    assert "_CONDOR_JOB_AD" in wrapper
    assert "expected_output_archive" in wrapper
    assert "build_attempt_archive.py" in wrapper
    assert "--execution-authorization" in wrapper


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            "execution_id,job_path,job_file_sha256,authorization_path,"
            "authorization_file_sha256,cpus,memory_mb,disk_mb,"
            "max_runtime_seconds",
            "execution_id,job_path,job_file_sha256,authorization_path,"
            "authorization_file_sha256,request_cpus,memory_mb,disk_mb,"
            "max_runtime_seconds",
        ),
        ("request_cpus = $(cpus)", "request_cpus = $(request_cpus)"),
    ),
)
def test_submit_contract_rejects_request_prefixed_item_variables(
    contract: ModuleType,
    before: str,
    after: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    assert before in submit
    tampered = submit.replace(before, after, 1)

    with pytest.raises(contract.ActivationContractError):
        contract.validate_submit_text(tampered)


@pytest.mark.parametrize(
    ("before", "after"),
    (
        ("max_materialize = 4", "max_materialize = 5"),
        ("leave_in_queue = True", "leave_in_queue = False"),
    ),
)
def test_submit_contract_rejects_factory_policy_drift(
    contract: ModuleType,
    before: str,
    after: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    assert before in submit
    with pytest.raises(contract.ActivationContractError):
        contract.validate_submit_text(submit.replace(before, after, 1))


@pytest.mark.parametrize(
    "competing_line",
    (
        "max_materialize = 48",
        "leave_in_queue = False",
        "max_idle = 4",
        "+JobMaterializeLimit = 4",
        "+JobMaterializeMaxIdle = 4",
        "MY.JobMaterializeLimit = 4",
    ),
)
def test_submit_contract_rejects_competing_factory_assignments(
    contract: ModuleType,
    competing_line: str,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    with pytest.raises(
        contract.ActivationContractError,
        match="[Ff]actory|retain",
    ):
        contract.validate_submit_text(f"{submit}\n{competing_line}\n")


def test_expanded_dry_run_projection_removes_only_factory_limit(
    contract: ModuleType,
) -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    projection = contract.expanded_dry_run_submit_text(submit)

    assert "max_materialize" not in projection.lower()
    assert "leave_in_queue = True" in projection
    assert projection == submit.replace("max_materialize = 4\n", "", 1)


def test_remote_dry_run_contract_accepts_exactly_48_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _dry_run_text(activation_result["manifest"]["executions"])
    receipt = contract.validate_remote_dry_run_text(
        text,
        executions=activation_result["manifest"]["executions"],
    )

    assert receipt["status"] == "passed"
    assert receipt["dry_run_kind"] == (
        "expanded_nonfactory_projection_v1"
    )
    assert receipt["ad_count"] == 48
    assert receipt["proc_ids"] == list(range(48))
    assert receipt["observed_leave_in_queue"] is True
    assert len(receipt["resources"]) == 48
    assert all(
        row["RequestDisk"]
        == (
            activation_result["manifest"]["executions"][index][
                "resources"
            ]["request_disk_mb"]
            * 1024
        )
        for index, row in enumerate(receipt["resources"])
    )


def test_remote_dry_run_contract_accepts_condor_25_delta_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _condor_delta_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    receipt = contract.validate_remote_dry_run_text(
        text,
        executions=activation_result["manifest"]["executions"],
    )

    assert receipt["status"] == "passed"
    assert receipt["ad_count"] == 48
    assert receipt["proc_ids"] == list(range(48))
    assert len(receipt["resources"]) == 48
    blocks = text.strip().split("\n\n")
    assert "RequestMemory=57344" in blocks[1]
    assert "RequestMemory" not in blocks[2]
    assert receipt["resources"][2]["RequestMemory"] == 49152


def test_remote_dry_run_contract_rejects_omitted_delta_override(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _condor_delta_dry_run_text(
        activation_result["manifest"]["executions"]
    )
    assert "ProcId=1\nRequestMemory=57344" in text

    with pytest.raises(
        contract.ActivationContractError,
        match="resource drift",
    ):
        contract.validate_remote_dry_run_text(
            text.replace(
                "ProcId=1\nRequestMemory=57344",
                "ProcId=1",
                1,
            ),
            executions=activation_result["manifest"]["executions"],
        )


def test_remote_dry_run_contract_rejects_retention_drift(
    contract: ModuleType,
    activation_result: dict[str, Any],
) -> None:
    text = _dry_run_text(activation_result["manifest"]["executions"])
    with pytest.raises(
        contract.ActivationContractError,
        match="completion-retention",
    ):
        contract.validate_remote_dry_run_text(
            text.replace(
                "LeaveJobInQueue = true",
                "LeaveJobInQueue = false",
                1,
            ),
            executions=activation_result["manifest"]["executions"],
        )


def test_factory_dry_run_contract_accepts_cluster_ad(
    contract: ModuleType,
) -> None:
    receipt = contract.validate_remote_factory_dry_run_text(
        _factory_dry_run_text()
    )

    assert receipt["status"] == "passed"
    assert receipt["dry_run_kind"] == "factory_cluster_ad_v1"
    assert receipt["cluster_id"] == 1
    assert receipt["observed_leave_in_queue"] is True
    assert receipt["live_factory_query_required"] is True
    assert receipt["live_factory_expected_attributes"] == {
        "JobMaterializeLimit": 4,
        "TotalSubmitProcs": 48,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("ClusterId=1", "ProcId=0"),
        lambda value: value.replace(
            f'JobBatchName="{EXPECTED_BATCH_NAME}"',
            'JobBatchName="wrong-batch"',
        ),
        lambda value: value.replace(
            "LeaveJobInQueue=true",
            "LeaveJobInQueue=false",
        ),
    ),
)
def test_factory_dry_run_contract_rejects_drift(
    contract: ModuleType,
    mutation: Any,
) -> None:
    with pytest.raises(contract.ActivationContractError):
        contract.validate_remote_factory_dry_run_text(
            mutation(_factory_dry_run_text())
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace(
            "ProcId = 47", "ProcId = 46", 1
        ),
        lambda value: value.replace(
            "RequestDisk = ", "RequestDisk = 1", 1
        ),
        lambda value: value.replace(
            "RequestMemory = ", "Requestmemory_mb = ", 1
        ),
    ),
)
def test_remote_dry_run_contract_rejects_malformed_ads(
    contract: ModuleType,
    activation_result: dict[str, Any],
    mutation: Any,
) -> None:
    text = _dry_run_text(activation_result["manifest"]["executions"])

    with pytest.raises(contract.ActivationContractError):
        contract.validate_remote_dry_run_text(
            mutation(text),
            executions=activation_result["manifest"]["executions"],
        )


def test_activation_read_only_validator_passes(
    activation_result: dict[str, Any],
    tmp_path: Path,
) -> None:
    expanded_path = tmp_path / "factorial48.expanded-dry-run.classads"
    expanded_path.write_text(
        _dry_run_text(activation_result["manifest"]["executions"]),
        encoding="utf-8",
    )
    factory_path = tmp_path / "factorial48.factory-dry-run.classad"
    factory_path.write_text(_factory_dry_run_text(), encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [
            sys.executable,
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
    assert payload["direct_execution_count"] == 48
    assert payload["submission_state"] == "authorized_not_submitted"
    assert payload["remote_stage"] is False
    assert payload["condor_submit"] is False
    assert payload["submitted"] is False
    assert payload["remote_dry_run_validation"]["status"] == "passed"
    expanded = payload["remote_dry_run_validation"][
        "expanded_nonfactory_projection"
    ]
    factory = payload["remote_dry_run_validation"]["factory_cluster_ad"]
    assert expanded["ad_count"] == 48
    assert expanded["proc_ids"] == list(range(48))
    assert expanded["observed_leave_in_queue"] is True
    assert factory["cluster_id"] == 1
    assert factory["observed_leave_in_queue"] is True
    assert payload["remote_dry_run_validation"][
        "live_factory_expected_attributes"
    ] == {
        "JobMaterializeLimit": 4,
        "TotalSubmitProcs": 48,
    }
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))
