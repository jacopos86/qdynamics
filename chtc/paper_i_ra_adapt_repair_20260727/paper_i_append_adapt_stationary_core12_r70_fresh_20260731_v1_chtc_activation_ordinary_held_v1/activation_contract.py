#!/usr/bin/env python3
"""Fail-closed contract for the Append-ADAPT r70 held-start activation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True


PACKAGE_ID = (
    "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_held_v1"
ACTIVATION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_held_activation_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_execution_authorization_v1"
)
BATCH_NAME = (
    "paper-i-append-adapt-stationary-core12-r70-fresh-"
    "20260731-v1-ordinary-held-v1"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "eea38b59e60d727281dc3bdaf6d2efa7880f3f49375ce49e61134fbb35a566ea"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "334fb630c061d205d61554dca4b3e4f734edf5af394eda1ddb9c994869efefee"
)
EXECUTION_PLAN_CANONICAL_SHA256 = (
    "8289b35f84220ac5704e1eff4349f0e243c06a53136766c6a386e63657f34dc8"
)
EXECUTION_PLAN_FILE_SHA256 = (
    "076e5a31e16b4a743d56d4c860db890437f53237f6c1ce823b91ef182af7ee9b"
)
HORIZON_AUDIT_CANONICAL_SHA256 = (
    "7f6bcf0cc8e12f69e77fdf10260f3c854b2afee65d6f66e268432355ad15f74e"
)
SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)
DIRECT_EXECUTION_COUNT = 12
QUEUE_VARIABLES = (
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
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "execute_authorized_job.sh",
    "build_attempt_archive.py",
    "submit.sub",
)
GENERATED_FILES = (
    "execution_authorization.json",
    "queue.tsv",
    "activation_manifest.json",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ActivationContractError(ValueError):
    """Raised when the activation or its sealed package drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise ActivationContractError("Digest input already has sha256.")
    result["sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ActivationContractError(f"{label} must be an object.")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("sha256")
    body = dict(payload)
    body.pop("sha256", None)
    if (
        not isinstance(expected, str)
        or not _HEX64.fullmatch(expected)
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != expected
    ):
        raise ActivationContractError(f"{label} digest drifted.")


def repo_root_from_script(script: str | Path) -> Path:
    root = Path(script).resolve().parents[3]
    if not (root / "AGENTS.md").is_file():
        raise ActivationContractError("Repository root could not be resolved.")
    return root


def activation_relative() -> PurePosixPath:
    return PurePosixPath(
        "chtc/paper_i_ra_adapt_repair_20260727"
    ) / ACTIVATION_ID


def package_relative() -> PurePosixPath:
    return PurePosixPath(
        "chtc/paper_i_ra_adapt_repair_20260727"
    ) / PACKAGE_ID


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe bound file: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=path.name)
    verify_self_digest(payload, label=path.name)
    binding = file_binding(path, relative_to=relative_to)
    binding["canonical_sha256"] = payload["sha256"]
    return binding


def _load_package_contract(package_dir: Path) -> ModuleType:
    path = package_dir / "package_contract.py"
    spec = importlib.util.spec_from_file_location(
        "append_r70_sealed_package_contract", path
    )
    if spec is None or spec.loader is None:
        raise ActivationContractError("Package contract cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sealed_package(repo_root: Path) -> dict[str, Any]:
    package_dir = repo_root / package_relative()
    contract = _load_package_contract(package_dir)
    result = contract.validate_package(
        package_dir=package_dir,
        full_archive_scan=True,
        full_anchor_scan=False,
    )
    manifest_path = package_dir / "package_manifest.json"
    plan_path = package_dir / "execution_plan.json"
    manifest = load_json(manifest_path, label="package manifest")
    plan = load_json(plan_path, label="execution plan")
    audit = load_json(
        package_dir / "horizon_delta_audit.json",
        label="horizon audit",
    )
    if (
        result.get("status") != "passed"
        or manifest.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or sha256_file(manifest_path) != PACKAGE_MANIFEST_FILE_SHA256
        or plan.get("sha256") != EXECUTION_PLAN_CANONICAL_SHA256
        or sha256_file(plan_path) != EXECUTION_PLAN_FILE_SHA256
        or audit.get("sha256") != HORIZON_AUDIT_CANONICAL_SHA256
        or audit.get("status") != "pass"
        or sha256_file(package_dir / "source_r50_locked.tar.gz")
        != SOURCE_ARCHIVE_SHA256
        or manifest.get("run_class") != "paper_facing"
        or manifest.get("direct_execution_count") != DIRECT_EXECUTION_COUNT
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_state") != "not_submitted"
    ):
        raise ActivationContractError("Sealed package authority drifted.")
    return {"manifest": manifest, "plan": plan, "audit": audit}


def validate_submit_text(text: str) -> None:
    required = (
        f'+JobBatchName = "{BATCH_NAME}"',
        '+HolsteinLifecycleMode = "ordinary_held_exact_proc_release_v1"',
        "hold = True",
        "periodic_release = False",
        "leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)",
        "stream_output = False",
        "stream_error = False",
        "queue " + ",".join(QUEUE_VARIABLES) + " from ",
    )
    if any(token not in text for token in required):
        raise ActivationContractError("Submit descriptor contract drifted.")
    lowered = text.lower()
    if "max_materialize" in lowered or "max_idle" in lowered:
        raise ActivationContractError(
            "Ordinary held activation must not be a factory."
        )
    if re.search(r"(?im)^\s*periodic_release\s*=\s*true\s*$", text):
        raise ActivationContractError("Automatic release is forbidden.")


def _control_bindings(activation_dir: Path) -> list[dict[str, Any]]:
    return [
        file_binding(activation_dir / name, relative_to=activation_dir)
        for name in CONTROL_FILES
    ]


def _parse_queue(path: Path) -> list[list[str]]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError("Activation queue is unsafe.")
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    if (
        len(rows) != DIRECT_EXECUTION_COUNT
        or any(len(row) != len(QUEUE_VARIABLES) for row in rows)
    ):
        raise ActivationContractError("Activation queue shape drifted.")
    return rows


def quota_release_contract() -> dict[str, Any]:
    return {
        "initial_state": "all_rows_submitted_on_hold",
        "automatic_release": False,
        "release_scope": "one_exact_cluster.proc_only",
        "next_release_requires": [
            "current_hard_quota_headroom",
            "conservative_next_archive_estimate_plus_margin",
            "previous_archive_fetched_and_verified",
            (
                "previous_remote_archive_preserved_or_exact_"
                "deleted_after_verification"
            ),
            "quota_rechecked_after_transfer_handle_close",
        ],
    }


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = validate_sealed_package(root)
    activation_dir = root / activation_relative()
    if Path(__file__).resolve().parent != activation_dir:
        raise ActivationContractError("Activation directory drifted.")
    submit_text = (activation_dir / "submit.sub").read_text(encoding="utf-8")
    validate_submit_text(submit_text)

    controls = _control_bindings(activation_dir)
    control_sha = hashlib.sha256(canonical_json_bytes(controls)).hexdigest()
    authorization_path = activation_dir / "execution_authorization.json"
    authorization = load_json(
        authorization_path, label="execution authorization"
    )
    verify_self_digest(authorization, label="execution authorization")
    plan_ids = package["plan"].get("execution_ids")
    fixed_authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "passed",
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "campaign_id": package["manifest"]["campaign_id"],
        "package_manifest_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
        "horizon_delta_audit_sha256": HORIZON_AUDIT_CANONICAL_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "activation_control_plane_sha256": control_sha,
        "remote_image_path": IMAGE_PATH,
        "remote_image_sha256": IMAGE_SHA256,
        "remote_image_byte_verification_passed": True,
        "authorized_execution_ids": plan_ids,
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    for key, value in fixed_authorization.items():
        if authorization.get(key) != value:
            raise ActivationContractError(
                f"Execution authorization drifted at {key}."
            )
    expected_authorization_keys = set(fixed_authorization) | {
        "authorization_id",
        "authorized_utc",
        "remote_image_verified_utc",
        "sha256",
    }
    if set(authorization) != expected_authorization_keys:
        raise ActivationContractError(
            "Execution authorization field closure drifted."
        )
    if authorization.get("authorization_id") != (
        f"{ACTIVATION_ID}__all12"
    ):
        raise ActivationContractError("Authorization identity drifted.")
    if not authorization.get("authorized_utc") or not authorization.get(
        "remote_image_verified_utc"
    ):
        raise ActivationContractError("Authorization timestamps are absent.")

    queue_path = activation_dir / "queue.tsv"
    rows = _parse_queue(queue_path)
    package_dir = root / package_relative()
    authorization_binding = file_binding(
        authorization_path, relative_to=root
    )
    plan_rows = package["plan"].get("direct_executions")
    if not isinstance(plan_rows, list):
        raise ActivationContractError("Execution-plan rows are absent.")
    expected_executions: list[dict[str, Any]] = []
    for index, (queue_row, planned) in enumerate(
        zip(rows, plan_rows, strict=True)
    ):
        if not isinstance(planned, Mapping):
            raise ActivationContractError("Execution-plan row is malformed.")
        execution_id = str(planned.get("execution_id"))
        job_path = package_dir / str(planned.get("job_spec_path"))
        job = load_json(job_path, label=f"job {index}")
        verify_self_digest(job, label=f"job {index}")
        resources = job.get("resources")
        if not isinstance(resources, Mapping):
            raise ActivationContractError("Job resources are absent.")
        expected = [
            execution_id,
            job_path.relative_to(root).as_posix(),
            sha256_file(job_path),
            authorization_path.relative_to(root).as_posix(),
            authorization_binding["sha256"],
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if queue_row != expected:
            raise ActivationContractError(
                f"Queue row drifted at index {index}."
            )
        expected_executions.append(
            {
                "queue_index": index,
                "execution_id": execution_id,
                "job": json_binding(job_path, relative_to=root),
                "resources": {
                    "request_cpus": resources["request_cpus"],
                    "request_memory_mb": resources["request_memory_mb"],
                    "request_disk_mb": resources["request_disk_mb"],
                    "max_runtime_seconds": resources[
                        "max_runtime_seconds"
                    ],
                },
            }
        )

    manifest = load_json(
        activation_dir / "activation_manifest.json",
        label="activation manifest",
    )
    verify_self_digest(manifest, label="activation manifest")
    fixed_manifest = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "campaign_id": package["manifest"]["campaign_id"],
        "batch_name": BATCH_NAME,
        "run_class": "paper_facing",
        "execution_target": "chtc",
        "direct_execution_count": DIRECT_EXECUTION_COUNT,
        "queue_variables": list(QUEUE_VARIABLES),
        "activation_control_plane_sha256": control_sha,
        "operational_mode": "ordinary_held_exact_proc_release_v1",
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adopted": False,
    }
    for key, value in fixed_manifest.items():
        if manifest.get(key) != value:
            raise ActivationContractError(
                f"Activation manifest drifted at {key}."
            )
    expected_manifest_keys = set(fixed_manifest) | {
        "authorized_utc",
        "sealed_package",
        "remote_image",
        "control_plane",
        "execution_authorization",
        "executions",
        "queue",
        "quota_release_contract",
        "sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise ActivationContractError(
            "Activation manifest field closure drifted."
        )
    if manifest.get("control_plane") != controls:
        raise ActivationContractError("Control-plane bindings drifted.")
    if manifest.get("authorized_utc") != authorization["authorized_utc"]:
        raise ActivationContractError("Activation authorization time drifted.")
    if manifest.get("quota_release_contract") != quota_release_contract():
        raise ActivationContractError("Quota-release contract drifted.")
    if manifest.get("execution_authorization") != json_binding(
        authorization_path, relative_to=activation_dir
    ):
        raise ActivationContractError("Authorization binding drifted.")
    if manifest.get("queue") != file_binding(
        queue_path, relative_to=activation_dir
    ):
        raise ActivationContractError("Queue binding drifted.")
    if manifest.get("executions") != expected_executions:
        raise ActivationContractError("Execution bindings drifted.")
    if manifest.get("sealed_package") != {
        "path": package_relative().as_posix(),
        "manifest_canonical_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
        "manifest_file_sha256": PACKAGE_MANIFEST_FILE_SHA256,
        "execution_plan_canonical_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
        "execution_plan_file_sha256": EXECUTION_PLAN_FILE_SHA256,
        "horizon_delta_audit_sha256": HORIZON_AUDIT_CANONICAL_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }:
        raise ActivationContractError("Sealed package binding drifted.")
    if manifest.get("remote_image") != {
        "path": IMAGE_PATH,
        "sha256": IMAGE_SHA256,
        "byte_verification_passed": True,
        "verified_utc": authorization["remote_image_verified_utc"],
    }:
        raise ActivationContractError("Remote image binding drifted.")
    actual_files = {
        path.relative_to(activation_dir).as_posix()
        for path in activation_dir.rglob("*")
        if path.is_file()
    }
    if actual_files != set(CONTROL_FILES) | set(GENERATED_FILES):
        raise ActivationContractError("Activation file closure drifted.")
    return {
        "status": "passed",
        "activation_id": ACTIVATION_ID,
        "activation_manifest_sha256": manifest["sha256"],
        "batch_name": BATCH_NAME,
        "direct_execution_count": DIRECT_EXECUTION_COUNT,
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "operational_mode": "ordinary_held_exact_proc_release_v1",
    }


__all__ = [
    "ACTIVATION_ID",
    "ACTIVATION_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "BATCH_NAME",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_CANONICAL_SHA256",
    "EXECUTION_PLAN_FILE_SHA256",
    "GENERATED_FILES",
    "HORIZON_AUDIT_CANONICAL_SHA256",
    "IMAGE_PATH",
    "IMAGE_SHA256",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_CANONICAL_SHA256",
    "PACKAGE_MANIFEST_FILE_SHA256",
    "QUEUE_VARIABLES",
    "SOURCE_ARCHIVE_SHA256",
    "ActivationContractError",
    "activation_relative",
    "canonical_json_bytes",
    "digested",
    "file_binding",
    "json_binding",
    "load_json",
    "package_relative",
    "quota_release_contract",
    "repo_root_from_script",
    "sha256_file",
    "validate_activation",
    "validate_sealed_package",
    "validate_submit_text",
    "verify_self_digest",
]
