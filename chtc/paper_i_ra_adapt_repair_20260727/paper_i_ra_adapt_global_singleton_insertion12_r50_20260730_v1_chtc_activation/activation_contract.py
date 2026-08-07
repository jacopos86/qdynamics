#!/usr/bin/env python3
"""Fail-closed activation contract for the 12-cell singleton comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc"
)
ACTIVATION_ID = (
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc_activation_v1"
)
BATCH_NAME = (
    "paper-i-ra-adapt-global-singleton-insertion12-r50-20260730-v1"
)
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc"
)
ACTIVATION_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc_activation"
)
RUNTIME_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_insertion12_"
    "r50_20260730_v1_chtc_runtime"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)

PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "a7e8663e0b9daa3b7589652179e8e6ec6ebb7d4ad47925a294beed232b268940"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "b77b21f39e38a27b633b5c6b02e358a8c00cb8b02d0863d0f54922f3c4c7a838"
)
EXECUTION_PLAN_CANONICAL_SHA256 = (
    "3a2be2ffc22efc3896c80f8f00b65baeed1595353ad47c51a30f2bea7df0b85a"
)
EXECUTION_PLAN_FILE_SHA256 = (
    "c72bcf2c30ebcd678ef5b57cd19a2c5eaff5a2e0d3f83dfe0745be956b48bc1c"
)
SOURCE_ARCHIVE_SHA256 = (
    "2705bc4c424b9d9e4b116d2e3fe061359c3704ba2f504ac113e35d15c23411ac"
)
CALIBRATION_CANONICAL_SHA256 = (
    "98dec786b814a68ac7517325004d702e04123c048b55b9ccb0363100be94403b"
)
CALIBRATION_FILE_SHA256 = (
    "1f5492655411dcf6e00090fbb4a41c147c3a3bbdca9bd621f16be7ba0c2cee20"
)

JOB_SCHEMA = "paper_i_ra_global_singleton_insertion_job_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_"
    "execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_"
    "activation_manifest_v1"
)
REMOTE_DRY_RUN_SCHEMA = (
    "paper_i_ra_global_singleton_insertion12_"
    "dual_remote_dry_run_validation_v2"
)
EXPANDED_DRY_RUN_KIND = "expanded_nonfactory_projection_v1"
FACTORY_DRY_RUN_KIND = "factory_cluster_ad_v1"
CALIBRATION_SCHEMA = "plateau_open_domain_calibration_v1"
CALIBRATION_RECEIPT_NAME = (
    "plateau_open_domain_calibration_receipt.json"
)
DIRECT_EXECUTION_COUNT = 12
MAX_MATERIALIZE = 1
LEAVE_IN_QUEUE = True
RESOURCE_STATUS = "provisional_not_demonstrated"
JOB_RESOURCE_STATUS = (
    "provisional_not_demonstrated_by_bounded_calibration"
)
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
JOB_RESOURCE_KEYS = (
    "request_cpus",
    "request_memory_mb",
    "request_disk_mb",
    "max_runtime_seconds",
)
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "build_attempt_archive.py",
    "execute_authorized_job.sh",
    "submit.sub",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*\+?([A-Za-z][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*;?\s*$"
)
_FORBIDDEN_RESOURCE_PATTERNS = (
    re.compile(r"\bRequestmemory_mb\b", re.IGNORECASE),
    re.compile(r"\bRequestdisk_mb\b", re.IGNORECASE),
    re.compile(r"\bTARGET\.memory_mb\b", re.IGNORECASE),
    re.compile(r"\bTARGET\.disk_mb\b", re.IGNORECASE),
)


class ActivationContractError(ValueError):
    """Raised when package, activation, or dry-run state drifts."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"{label} is missing or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationContractError(f"{label} is unreadable.") from exc
    if not isinstance(value, dict):
        raise ActivationContractError(f"{label} is not a mapping.")
    return value


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("sha256") != canonical_sha256(payload):
        raise ActivationContractError(f"{label} self-digest drifted.")


def safe_relative_path(value: Any, *, label: str) -> Path:
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise ActivationContractError(f"Unsafe {label}: {value}")
    return Path(*path.parts)


def repo_root_from_script(script: str | Path) -> Path:
    path = Path(script).resolve()
    for parent in (path.parent, *path.parents):
        if (
            (parent / "AGENTS.md").is_file()
            and (parent / "pipelines").is_dir()
            and (parent / "chtc").is_dir()
        ):
            return parent
    raise ActivationContractError("Repository root could not be resolved.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe binding source: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    verify_self_digest(payload, label=f"{path.name} binding")
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": payload["sha256"],
    }


def _verify_file_binding(
    binding: Mapping[str, Any],
    *,
    base: Path,
    label: str,
    json_payload: bool = False,
) -> Path:
    path = base / safe_relative_path(binding.get("path"), label=label)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != binding.get("sha256")
        or path.stat().st_size != int(binding.get("size_bytes", -1))
    ):
        raise ActivationContractError(f"{label} file binding drifted.")
    if json_payload:
        payload = load_json(path, label=label)
        verify_self_digest(payload, label=label)
        if payload["sha256"] != binding.get("canonical_sha256"):
            raise ActivationContractError(
                f"{label} canonical binding drifted."
            )
    return path


def _require_pins() -> None:
    pins = {
        "package manifest": PACKAGE_MANIFEST_CANONICAL_SHA256,
        "package manifest file": PACKAGE_MANIFEST_FILE_SHA256,
        "execution plan": EXECUTION_PLAN_CANONICAL_SHA256,
        "execution plan file": EXECUTION_PLAN_FILE_SHA256,
        "source archive": SOURCE_ARCHIVE_SHA256,
        "calibration": CALIBRATION_CANONICAL_SHA256,
        "calibration file": CALIBRATION_FILE_SHA256,
        "remote image": REMOTE_IMAGE_SHA256,
    }
    for label, value in pins.items():
        if _SHA256_RE.fullmatch(value) is None:
            raise ActivationContractError(
                f"{label} is not pinned to a SHA-256 digest."
            )


def _package_inventory(package_dir: Path) -> set[str]:
    if not package_dir.is_dir() or package_dir.is_symlink():
        raise ActivationContractError(
            "Global-singleton package is unavailable."
        )
    paths = tuple(package_dir.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ActivationContractError(
            "Global-singleton package contains a symlink."
        )
    return {
        path.relative_to(package_dir).as_posix()
        for path in paths
        if path.is_file()
    }


def _run_package_validator(
    repo_root: Path,
    package_dir: Path,
) -> dict[str, Any]:
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
            str(package_dir / "validate_package.py"),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ActivationContractError(
            "Inert package standalone validation failed"
            + (f": {detail}" if detail else ".")
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ActivationContractError(
            "Inert package validator emitted invalid JSON."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "passed"
        or payload.get("package_id") != PACKAGE_ID
        or payload.get("direct_execution_count")
        != DIRECT_EXECUTION_COUNT
        or payload.get("resource_status") != RESOURCE_STATUS
        or payload.get("execution_authorized") is not False
        or payload.get("submission_authorized") is not False
        or payload.get("remote_stage") is not False
        or payload.get("condor_submit") is not False
    ):
        raise ActivationContractError(
            "Inert package standalone validation receipt drifted."
        )
    return payload


def validate_provisional_calibration_payload(
    calibration: Mapping[str, Any],
) -> None:
    verify_self_digest(
        calibration,
        label="open-domain calibration",
    )
    domain = calibration.get("open_domain_receipt")
    resources = calibration.get("resource_observation")
    if (
        calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("status") != "passed"
        or calibration.get("package_id") != PACKAGE_ID
        or calibration.get("synthetic_trigger_only") is not True
        or calibration.get("scientific_result") is not False
        or calibration.get("execution_evidence") is not False
        or calibration.get("checkpoint_emitted") is not False
        or calibration.get("result_promotable") is not False
        or calibration.get("nph") != 7
        or calibration.get("candidate_count") != 6508
        or calibration.get("requested_positions") != [0, 1]
        or calibration.get(
            "precollapse_candidate_position_pair_count"
        )
        != 13_016
        or not isinstance(domain, Mapping)
        or domain.get("schema")
        != "insertion_commutation_plateau_round_policy_v1"
        or domain.get("policy")
        != "insertion_commutation_plateau_v1"
        or domain.get("domain_open") is not True
        or domain.get("domain_state") != "open"
        or domain.get("effective_insertion_mode")
        != "full_commutation_reduced"
        or domain.get("requested_positions") != [0, 1]
        or int(domain.get("candidate_count", -1)) != 6508
        or int(domain.get("retained_representative_count", -1))
        + int(domain.get("collapsed_position_count", -1))
        != 13_016
        or not isinstance(resources, Mapping)
        or int(resources.get("peak_rss_bytes", 0)) <= 0
        or float(resources.get("elapsed_seconds", 0.0)) <= 0.0
        or calibration.get("package_resources_demonstrated") is not False
        or calibration.get("package_resource_status")
        != RESOURCE_STATUS
    ):
        raise ActivationContractError(
            "Required provisional calibration receipt is invalid."
        )


def validate_sealed_package(
    repo_root: str | Path,
) -> dict[str, Any]:
    _require_pins()
    root = Path(repo_root).resolve()
    package_dir = root / PACKAGE_RELATIVE
    observed_inventory = _package_inventory(package_dir)
    validator_receipt = _run_package_validator(root, package_dir)

    manifest_path = package_dir / "package_manifest.json"
    plan_path = package_dir / "execution_plan.json"
    archive_path = package_dir / "source_locked.tar.gz"
    calibration_path = package_dir / CALIBRATION_RECEIPT_NAME
    manifest = load_json(manifest_path, label="package manifest")
    plan = load_json(plan_path, label="execution plan")
    calibration = load_json(
        calibration_path, label="open-domain calibration"
    )
    for payload, label in (
        (manifest, "package manifest"),
        (plan, "execution plan"),
        (calibration, "open-domain calibration"),
    ):
        verify_self_digest(payload, label=label)

    observed_pins = {
        "package manifest canonical": manifest.get("sha256"),
        "package manifest file": sha256_file(manifest_path),
        "execution plan canonical": plan.get("sha256"),
        "execution plan file": sha256_file(plan_path),
        "source archive": sha256_file(archive_path),
        "calibration canonical": calibration.get("sha256"),
        "calibration file": sha256_file(calibration_path),
    }
    expected_pins = {
        "package manifest canonical": (
            PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        "package manifest file": PACKAGE_MANIFEST_FILE_SHA256,
        "execution plan canonical": EXECUTION_PLAN_CANONICAL_SHA256,
        "execution plan file": EXECUTION_PLAN_FILE_SHA256,
        "source archive": SOURCE_ARCHIVE_SHA256,
        "calibration canonical": CALIBRATION_CANONICAL_SHA256,
        "calibration file": CALIBRATION_FILE_SHA256,
    }
    if observed_pins != expected_pins:
        raise ActivationContractError(
            "Sealed package byte authority drifted."
        )
    if (
        manifest.get("package_id") != PACKAGE_ID
        or plan.get("package_id") != PACKAGE_ID
        or manifest.get("status") != "passed"
        or manifest.get("direct_execution_count")
        != DIRECT_EXECUTION_COUNT
        or plan.get("direct_execution_count")
        != DIRECT_EXECUTION_COUNT
        or manifest.get("resource_status") != RESOURCE_STATUS
        or plan.get("resource_status") != RESOURCE_STATUS
        or manifest.get("authority_overlay_present") is not False
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_state") != "not_submitted"
        or manifest.get("submitted") is not False
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("submission_state") != "not_submitted"
        or plan.get("submitted") is not False
    ):
        raise ActivationContractError(
            "Sealed package inert envelope drifted."
        )
    validate_provisional_calibration_payload(calibration)
    calibration_binding = manifest.get("open_plateau_calibration")
    if (
        not isinstance(calibration_binding, Mapping)
        or calibration_binding.get("path")
        != CALIBRATION_RECEIPT_NAME
        or calibration_binding.get("sha256")
        != CALIBRATION_FILE_SHA256
        or calibration_binding.get("canonical_sha256")
        != CALIBRATION_CANONICAL_SHA256
    ):
        raise ActivationContractError(
            "Package lost its exact calibration binding."
        )

    source_archive = manifest.get("source_archive")
    plan_binding = manifest.get("execution_plan")
    smoke_binding = manifest.get("smoke_receipt")
    queue_binding = manifest.get("queue")
    jobs = manifest.get("jobs")
    controls = manifest.get("control_plane")
    if (
        not isinstance(source_archive, Mapping)
        or source_archive.get("path") != "source_locked.tar.gz"
        or source_archive.get("sha256") != SOURCE_ARCHIVE_SHA256
        or not isinstance(plan_binding, Mapping)
        or not isinstance(smoke_binding, Mapping)
        or not isinstance(queue_binding, Mapping)
        or not isinstance(jobs, list)
        or len(jobs) != DIRECT_EXECUTION_COUNT
        or not isinstance(controls, list)
        or not controls
    ):
        raise ActivationContractError(
            "Sealed package bindings are incomplete."
        )
    source_manifest_binding = source_archive.get("manifest")
    if not isinstance(source_manifest_binding, Mapping):
        raise ActivationContractError(
            "Source archive manifest binding is absent."
        )

    expected_inventory = {"package_manifest.json"}
    for index, binding in enumerate(controls):
        if not isinstance(binding, Mapping):
            raise ActivationContractError(
                "Package control binding is malformed."
            )
        path = _verify_file_binding(
            binding,
            base=package_dir,
            label=f"package control {index}",
        )
        expected_inventory.add(path.relative_to(package_dir).as_posix())
    for label, binding, is_json in (
        ("source archive", source_archive, False),
        ("source archive manifest", source_manifest_binding, True),
        ("execution plan", plan_binding, True),
        ("smoke receipt", smoke_binding, True),
        ("open-domain calibration", calibration_binding, True),
        ("package queue", queue_binding, False),
    ):
        path = _verify_file_binding(
            binding,
            base=package_dir,
            label=label,
            json_payload=is_json,
        )
        expected_inventory.add(path.relative_to(package_dir).as_posix())
    for index, binding in enumerate(jobs):
        if not isinstance(binding, Mapping):
            raise ActivationContractError(
                "Package job binding is malformed."
            )
        path = _verify_file_binding(
            binding,
            base=package_dir,
            label=f"package job {index}",
            json_payload=True,
        )
        expected_inventory.add(path.relative_to(package_dir).as_posix())
    if observed_inventory != expected_inventory:
        raise ActivationContractError(
            "Sealed package recursive closure drifted."
        )
    if (
        validator_receipt.get("package_manifest_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or validator_receipt.get("execution_plan_sha256")
        != EXECUTION_PLAN_CANONICAL_SHA256
        or validator_receipt.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or validator_receipt.get("open_plateau_calibration_sha256")
        != CALIBRATION_CANONICAL_SHA256
    ):
        raise ActivationContractError(
            "Standalone package validator lost pinned authority."
        )
    return {
        "package_dir": package_dir,
        "manifest": manifest,
        "plan": plan,
        "calibration": calibration,
        "validator_receipt": validator_receipt,
    }


def _validated_job_resources(job: Mapping[str, Any]) -> dict[str, Any]:
    raw = job.get("resources")
    expected_keys = {*JOB_RESOURCE_KEYS, "status"}
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ActivationContractError("Job resource contract drifted.")
    resources: dict[str, Any] = {}
    for name in JOB_RESOURCE_KEYS:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ActivationContractError(
                f"Job resource {name} must be a positive integer."
            )
        resources[name] = int(value)
    if raw.get("status") != JOB_RESOURCE_STATUS:
        raise ActivationContractError(
            "Job provisional resource status drifted."
        )
    resources["status"] = JOB_RESOURCE_STATUS
    return resources


def validate_authorization_payload(
    authorization: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verify_self_digest(authorization, label="execution authorization")
    execution_id = execution.get("execution_id")
    remote_image = manifest.get("remote_image")
    if not isinstance(remote_image, Mapping):
        raise ActivationContractError(
            "Activation remote-image binding is absent."
        )
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorization_id": f"{ACTIVATION_ID}__{execution_id}",
        "authorized_utc": manifest.get("authorized_utc"),
        "package_id": PACKAGE_ID,
        "activation_id": ACTIVATION_ID,
        "batch_name": BATCH_NAME,
        "execution_id": execution_id,
        "job_sha256": execution.get("job", {}).get(
            "canonical_sha256"
        ),
        "job_file_sha256": execution.get("job", {}).get("sha256"),
        "package_manifest_sha256": (
            PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        "execution_plan_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "open_plateau_calibration_sha256": (
            CALIBRATION_CANONICAL_SHA256
        ),
        "resource_status": RESOURCE_STATUS,
        "activation_control_plane_sha256": manifest.get(
            "activation_control_plane_sha256"
        ),
        "remote_image_path": REMOTE_IMAGE_PATH,
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "remote_image_byte_verification_passed": True,
        "remote_image_verified_utc": remote_image.get("verified_utc"),
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ActivationContractError(
            "Execution authorization binding drifted."
        )
    if not authorization.get("authorized_utc") or not authorization.get(
        "remote_image_verified_utc"
    ):
        raise ActivationContractError("Execution authorization is undated.")


def _parse_queue(path: Path) -> list[list[str]]:
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != DIRECT_EXECUTION_COUNT or any(
        len(row) != len(QUEUE_VARIABLES) for row in rows
    ):
        raise ActivationContractError("Activation queue is malformed.")
    return rows


def _submit_queue_variables(submit: str) -> tuple[str, ...]:
    matches = re.findall(
        r"(?im)^\s*queue\s+(.+?)\s+from\s+\S+\s*$",
        submit,
    )
    if len(matches) != 1:
        raise ActivationContractError(
            "Submit description must have exactly one queue-from statement."
        )
    variables = tuple(
        token.strip() for token in matches[0].split(",") if token.strip()
    )
    if variables != QUEUE_VARIABLES:
        raise ActivationContractError("Submit queue variables drifted.")
    if any(name.lower().startswith("request_") for name in variables):
        raise ActivationContractError(
            "Submit queue variables may not shadow request attributes."
        )
    return variables


def _submit_assignments(submit: str) -> list[tuple[str, str]]:
    assignments: list[tuple[str, str]] = []
    for line in submit.splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name, value = match.groups()
        assignments.append(
            (
                name.lower(),
                value.strip().rstrip(";").strip(),
            )
        )
    return assignments


def _require_exact_submit_assignment(
    submit: str,
    *,
    name: str,
    expected_value: str,
) -> None:
    observed = [
        value
        for assignment_name, value in _submit_assignments(submit)
        if assignment_name == name.lower()
    ]
    if observed != [expected_value]:
        raise ActivationContractError(
            f"Submit description must assign {name} exactly once "
            f"as {expected_value}."
        )


def validate_submit_text(submit: str) -> None:
    _submit_queue_variables(submit)
    if "$(request_" in submit.lower():
        raise ActivationContractError(
            "Submit description uses a request_-prefixed item variable."
        )
    _require_exact_submit_assignment(
        submit,
        name="max_materialize",
        expected_value=str(MAX_MATERIALIZE),
    )
    _require_exact_submit_assignment(
        submit,
        name="leave_in_queue",
        expected_value="True",
    )
    forbidden_factory_names = {
        "max_idle",
        "jobmaterializelimit",
        "jobmaterializemaxidle",
        "my.jobmaterializelimit",
        "my.jobmaterializemaxidle",
    }
    if any(
        name in forbidden_factory_names
        for name, _ in _submit_assignments(submit)
    ):
        raise ActivationContractError(
            "Submit description uses a competing factory-limit assignment."
        )
    expected_lines = {
        "request_cpus = $(cpus)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        "+MaxRuntime = $(max_runtime_seconds)",
        "when_to_transfer_output = ON_EXIT",
        "requirements = TARGET.HasSIF",
        f'+JobBatchName = "{BATCH_NAME}"',
    }
    observed_lines = {
        line.strip() for line in submit.splitlines() if line.strip()
    }
    if not expected_lines.issubset(observed_lines):
        raise ActivationContractError(
            "Submit resource or lifecycle assignments drifted."
        )
    if "ON_EXIT_OR_EVICT" in submit:
        raise ActivationContractError("Eviction output transfer is forbidden.")
    if any(
        pattern.search(submit)
        for pattern in _FORBIDDEN_RESOURCE_PATTERNS
    ):
        raise ActivationContractError(
            "Submit description uses a nonexistent resource attribute."
        )
    for required in (
        SOURCE_ARCHIVE_SHA256,
        REMOTE_IMAGE_PATH,
        REMOTE_IMAGE_SHA256,
        PACKAGE_RELATIVE.as_posix(),
        ACTIVATION_RELATIVE.as_posix(),
        RUNTIME_RELATIVE.as_posix(),
        "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz",
        "$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz",
    ):
        if required not in submit:
            raise ActivationContractError(
                f"Submit description lost binding: {required}"
            )


def expanded_dry_run_submit_text(submit: str) -> str:
    """Return the exact nonfactory projection used for resource dry-run."""

    validate_submit_text(submit)
    lines = submit.splitlines(keepends=True)
    matched_indexes: list[int] = []
    for index, line in enumerate(lines):
        match = _ASSIGNMENT_RE.match(line.rstrip("\r\n"))
        if match is not None and match.group(1).lower() == "max_materialize":
            matched_indexes.append(index)
    if len(matched_indexes) != 1:
        raise ActivationContractError(
            "Expanded dry-run projection requires exactly one "
            "max_materialize line."
        )
    removed_index = matched_indexes[0]
    projected = "".join(
        line for index, line in enumerate(lines) if index != removed_index
    )
    if any(
        name in {
            "max_materialize",
            "max_idle",
            "jobmaterializelimit",
            "jobmaterializemaxidle",
            "my.jobmaterializelimit",
            "my.jobmaterializemaxidle",
        }
        for name, _ in _submit_assignments(projected)
    ):
        raise ActivationContractError(
            "Expanded dry-run projection retained a factory limit."
        )
    _require_exact_submit_assignment(
        projected,
        name="leave_in_queue",
        expected_value="True",
    )
    _submit_queue_variables(projected)
    return projected


def remote_dry_run_contract() -> dict[str, Any]:
    return {
        "schema": REMOTE_DRY_RUN_SCHEMA,
        "required_before_condor_submit": True,
        "mode": "dual_factory_and_expanded_v1",
        "expanded_nonfactory_projection": {
            "kind": EXPANDED_DRY_RUN_KIND,
            "expected_ad_count": DIRECT_EXECUTION_COUNT,
            "expected_proc_ids": list(range(DIRECT_EXECUTION_COUNT)),
            "projection_removes_exactly": ["max_materialize"],
            "required_resource_attributes": [
                "RequestCpus",
                "RequestMemory",
                "RequestDisk",
                "MaxRuntime",
            ],
            "required_lifecycle_attributes": ["LeaveJobInQueue"],
        },
        "factory_cluster_ad": {
            "kind": FACTORY_DRY_RUN_KIND,
            "expected_cluster_ad_count": 1,
            "forbidden_attributes": ["ProcId"],
            "required_attributes": [
                "ClusterId",
                "JobBatchName",
                "LeaveJobInQueue",
            ],
        },
        "post_submit_factory_expectations": {
            "required": True,
            "observed_in_pre_submit_dry_run": False,
            "JobMaterializeLimit": MAX_MATERIALIZE,
            "TotalSubmitProcs": DIRECT_EXECUTION_COUNT,
        },
        "leave_in_queue": LEAVE_IN_QUEUE,
        "request_memory_unit": "MiB",
        "request_disk_unit": "KiB",
        "request_disk_conversion": (
            "job_request_disk_mb_times_1024_v1"
        ),
        "forbidden_attributes": [
            "Requestmemory_mb",
            "Requestdisk_mb",
            "TARGET.memory_mb",
            "TARGET.disk_mb",
        ],
    }


def _parse_assignment_block(block: str) -> dict[str, str]:
    ad: dict[str, str] = {}
    for line in block.splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name, value = match.groups()
        lowered = name.lower()
        if lowered in ad:
            raise ActivationContractError(
                f"Remote dry-run ad repeats attribute {name}."
            )
        ad[lowered] = value.strip().rstrip(";").strip()
    return ad


def _parse_expanded_classad_blocks(text: str) -> list[dict[str, str]]:
    bracketed = [
        match.group(1)
        for match in re.finditer(r"\[(.*?)\]", text, re.DOTALL)
        if re.search(r"(?im)^\s*ProcId\s*=", match.group(1))
    ]
    raw_blocks = (
        bracketed
        if bracketed
        else [
            block
            for block in re.split(r"\n\s*\n", text)
            if re.search(r"(?im)^\s*ProcId\s*=", block)
        ]
    )
    ads: list[dict[str, str]] = []
    baseline: dict[str, str] = {}
    inherited_names = {
        "requestcpus",
        "requestmemory",
        "requestdisk",
        "maxruntime",
        "jobbatchname",
        "leavejobinqueue",
    }
    for block in raw_blocks:
        ad = _parse_assignment_block(block)
        if "procid" in ad:
            if bracketed:
                ads.append(ad)
            else:
                # HTCondor 25.13 emits the first dry-run ad in full and
                # subsequent ads as deltas against that cluster baseline.
                # Each process overlays the baseline independently.
                if not baseline:
                    baseline = {
                        name: value
                        for name, value in ad.items()
                        if name in inherited_names
                    }
                    ads.append(ad)
                else:
                    ads.append({**baseline, **ad})
    return ads


def _parse_factory_cluster_ad(text: str) -> dict[str, str]:
    if re.search(r"(?im)^\s*\+?ProcId\s*=", text):
        raise ActivationContractError(
            "Factory dry-run cluster ad must not contain ProcId."
        )
    bracketed = [
        match.group(1)
        for match in re.finditer(r"\[(.*?)\]", text, re.DOTALL)
        if re.search(
            r"(?im)^\s*ClusterId\s*=",
            match.group(1),
        )
    ]
    raw_blocks = (
        bracketed
        if bracketed
        else [
            block
            for block in re.split(r"\n\s*\n", text)
            if block.strip()
        ]
    )
    parsed = [
        ad
        for block in raw_blocks
        if (ad := _parse_assignment_block(block))
    ]
    if len(parsed) != 1:
        raise ActivationContractError(
            "Factory dry-run must contain exactly one cluster ad."
        )
    cluster_ad = parsed[0]
    return cluster_ad


def _integer_ad_value(ad: Mapping[str, str], name: str) -> int:
    raw = ad.get(name.lower())
    if raw is None:
        raise ActivationContractError(
            f"Remote dry-run ad is missing {name}."
        )
    unquoted = raw.strip().strip('"')
    if re.fullmatch(r"[0-9]+", unquoted) is None:
        raise ActivationContractError(
            f"Remote dry-run {name} is not an integer."
        )
    return int(unquoted)


def _string_ad_value(ad: Mapping[str, str], name: str) -> str:
    raw = ad.get(name.lower())
    if raw is None:
        raise ActivationContractError(
            f"Remote dry-run ad is missing {name}."
        )
    return raw.strip().strip('"')


def _boolean_ad_value(ad: Mapping[str, str], name: str) -> bool:
    value = _string_ad_value(ad, name).lower()
    if value not in {"true", "false"}:
        raise ActivationContractError(
            f"Remote dry-run {name} is not a boolean."
        )
    return value == "true"


def _post_submit_factory_expectations() -> dict[str, Any]:
    return {
        "required": True,
        "observed_in_pre_submit_dry_run": False,
        "JobMaterializeLimit": MAX_MATERIALIZE,
        "TotalSubmitProcs": DIRECT_EXECUTION_COUNT,
    }


def validate_remote_expanded_dry_run_text(
    text: str,
    *,
    executions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if any(
        pattern.search(text)
        for pattern in _FORBIDDEN_RESOURCE_PATTERNS
    ):
        raise ActivationContractError(
            "Expanded dry-run materialized a nonexistent resource attribute."
        )
    ads = _parse_expanded_classad_blocks(text)
    if len(ads) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError(
            "Expanded dry-run must materialize exactly 12 ads."
        )
    if len(executions) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError(
            "Activation execution count drifted before dry-run validation."
        )

    rows: list[dict[str, Any]] = []
    proc_ids: list[int] = []
    for index, (ad, execution) in enumerate(
        zip(ads, executions)
    ):
        proc_id = _integer_ad_value(ad, "ProcId")
        proc_ids.append(proc_id)
        if proc_id != index:
            raise ActivationContractError(
                "Expanded dry-run ProcId order is not exactly 0..11."
            )
        resources = execution.get("resources")
        if not isinstance(resources, Mapping):
            raise ActivationContractError(
                "Activation execution has no resource contract."
            )
        expected = {
            "RequestCpus": int(resources["request_cpus"]),
            "RequestMemory": int(resources["request_memory_mb"]),
            "RequestDisk": int(resources["request_disk_mb"]) * 1024,
            "MaxRuntime": int(resources["max_runtime_seconds"]),
        }
        observed = {
            name: _integer_ad_value(ad, name) for name in expected
        }
        if observed != expected:
            raise ActivationContractError(
                f"Expanded dry-run resource drift for ProcId {proc_id}."
            )
        if _string_ad_value(ad, "JobBatchName") != BATCH_NAME:
            raise ActivationContractError(
                "Expanded dry-run batch identity drifted."
            )
        if _boolean_ad_value(ad, "LeaveJobInQueue") is not LEAVE_IN_QUEUE:
            raise ActivationContractError(
                "Expanded dry-run completion-retention policy drifted."
            )
        rows.append(
            {
                "proc_id": proc_id,
                "execution_id": execution["execution_id"],
                **observed,
            }
        )
    if proc_ids != list(range(DIRECT_EXECUTION_COUNT)):
        raise ActivationContractError(
            "Expanded dry-run ProcId closure drifted."
        )
    return digested(
        {
            "schema": REMOTE_DRY_RUN_SCHEMA,
            "kind": EXPANDED_DRY_RUN_KIND,
            "status": "passed",
            "ad_count": len(ads),
            "proc_ids": proc_ids,
            "batch_name": BATCH_NAME,
            "leave_in_queue": LEAVE_IN_QUEUE,
            "post_submit_factory_expectations": (
                _post_submit_factory_expectations()
            ),
            "classad_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "resources": rows,
        }
    )


def validate_remote_factory_dry_run_text(text: str) -> dict[str, Any]:
    if any(
        pattern.search(text)
        for pattern in _FORBIDDEN_RESOURCE_PATTERNS
    ):
        raise ActivationContractError(
            "Factory dry-run materialized a nonexistent resource attribute."
        )
    cluster_ad = _parse_factory_cluster_ad(text)
    cluster_id = _integer_ad_value(cluster_ad, "ClusterId")
    if _string_ad_value(cluster_ad, "JobBatchName") != BATCH_NAME:
        raise ActivationContractError(
            "Factory dry-run batch identity drifted."
        )
    leave_in_queue = _boolean_ad_value(
        cluster_ad,
        "LeaveJobInQueue",
    )
    if leave_in_queue is not LEAVE_IN_QUEUE:
        raise ActivationContractError(
            "Factory dry-run completion-retention policy drifted."
        )
    return digested(
        {
            "schema": REMOTE_DRY_RUN_SCHEMA,
            "kind": FACTORY_DRY_RUN_KIND,
            "status": "passed",
            "cluster_ad_count": 1,
            "cluster_id": cluster_id,
            "batch_name": BATCH_NAME,
            "leave_in_queue": leave_in_queue,
            "post_submit_factory_expectations": (
                _post_submit_factory_expectations()
            ),
            "classad_text_sha256": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
        }
    )


def validate_remote_dry_run_text(
    text: str,
    *,
    executions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compatibility alias for the expanded nonfactory projection."""

    return validate_remote_expanded_dry_run_text(
        text,
        executions=executions,
    )


def _read_remote_dry_run(path_value: str | Path, *, label: str) -> str:
    path = Path(path_value)
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"{label} is missing or unsafe.")
    return path.read_text(encoding="utf-8")


def validate_remote_expanded_dry_run(
    repo_root: str | Path,
    dry_run_path: str | Path,
) -> dict[str, Any]:
    result = validate_activation(repo_root)
    return validate_remote_expanded_dry_run_text(
        _read_remote_dry_run(
            dry_run_path,
            label="Expanded dry-run classad file",
        ),
        executions=result["manifest"]["executions"],
    )


def validate_remote_factory_dry_run(
    repo_root: str | Path,
    dry_run_path: str | Path,
) -> dict[str, Any]:
    validate_activation(repo_root)
    return validate_remote_factory_dry_run_text(
        _read_remote_dry_run(
            dry_run_path,
            label="Factory dry-run classad file",
        )
    )


def validate_remote_dry_run(
    repo_root: str | Path,
    dry_run_path: str | Path,
) -> dict[str, Any]:
    """Compatibility alias for the expanded nonfactory projection."""

    return validate_remote_expanded_dry_run(repo_root, dry_run_path)


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = validate_sealed_package(root)
    package_dir = package["package_dir"]
    package_manifest = package["manifest"]
    plan = package["plan"]
    activation_dir = root / ACTIVATION_RELATIVE

    manifest_path = activation_dir / "activation_manifest.json"
    manifest = load_json(manifest_path, label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    fixed = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "batch_name": BATCH_NAME,
        "campaign_id": package_manifest.get("campaign_id"),
        "run_class": package_manifest.get("run_class"),
        "execution_target": "chtc",
        "direct_execution_count": DIRECT_EXECUTION_COUNT,
        "resource_status": RESOURCE_STATUS,
        "queue_variables": list(QUEUE_VARIABLES),
        "remote_dry_run_validation_contract": remote_dry_run_contract(),
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adopted": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ActivationContractError("Activation state drifted.")
    if not manifest.get("authorized_utc"):
        raise ActivationContractError("Activation authorization is undated.")
    if "cluster_id" in manifest:
        raise ActivationContractError(
            "Pre-submit activation claims a scheduler cluster."
        )
    remote_image = manifest.get("remote_image")
    if (
        not isinstance(remote_image, Mapping)
        or remote_image.get("path") != REMOTE_IMAGE_PATH
        or remote_image.get("sha256") != REMOTE_IMAGE_SHA256
        or remote_image.get("byte_verification_passed") is not True
        or not remote_image.get("verified_utc")
    ):
        raise ActivationContractError("Remote-image binding drifted.")

    sealed_package = manifest.get("sealed_package")
    expected_package = {
        "path": PACKAGE_RELATIVE.as_posix(),
        "manifest_canonical_sha256": (
            PACKAGE_MANIFEST_CANONICAL_SHA256
        ),
        "manifest_file_sha256": PACKAGE_MANIFEST_FILE_SHA256,
        "execution_plan_canonical_sha256": (
            EXECUTION_PLAN_CANONICAL_SHA256
        ),
        "execution_plan_file_sha256": EXECUTION_PLAN_FILE_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "open_plateau_calibration_canonical_sha256": (
            CALIBRATION_CANONICAL_SHA256
        ),
        "open_plateau_calibration_file_sha256": (
            CALIBRATION_FILE_SHA256
        ),
        "resource_status": RESOURCE_STATUS,
    }
    if sealed_package != expected_package:
        raise ActivationContractError(
            "Activation lost exact sealed-package binding."
        )

    controls = manifest.get("control_plane")
    if not isinstance(controls, list) or [
        row.get("path") for row in controls if isinstance(row, Mapping)
    ] != list(CONTROL_FILES):
        raise ActivationContractError("Activation control plane drifted.")
    for row in controls:
        if not isinstance(row, Mapping):
            raise ActivationContractError(
                "Control-plane row is malformed."
            )
        _verify_file_binding(
            row,
            base=activation_dir,
            label=f"control {row.get('path')}",
        )
    control_digest = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    if control_digest != manifest.get(
        "activation_control_plane_sha256"
    ):
        raise ActivationContractError("Control-plane digest drifted.")

    submit = (activation_dir / "submit.sub").read_text(encoding="utf-8")
    validate_submit_text(submit)
    queue_binding = manifest.get("queue")
    if not isinstance(queue_binding, Mapping):
        raise ActivationContractError("Activation queue binding is absent.")
    queue_path = _verify_file_binding(
        queue_binding,
        base=activation_dir,
        label="activation queue",
    )
    queue_rows = _parse_queue(queue_path)

    executions = manifest.get("executions")
    package_jobs = package_manifest.get("jobs")
    plan_ids = plan.get("execution_ids")
    if (
        not isinstance(executions, list)
        or len(executions) != DIRECT_EXECUTION_COUNT
        or not isinstance(package_jobs, list)
        or len(package_jobs) != DIRECT_EXECUTION_COUNT
        or not isinstance(plan_ids, list)
        or len(plan_ids) != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError("Activation execution count drifted.")

    authorizations: list[dict[str, Any]] = []
    expected_files = {
        *(activation_dir / name for name in CONTROL_FILES),
        activation_dir / "activation_manifest.json",
        activation_dir / "queue.tsv",
    }
    observed_ids: list[str] = []
    for index, (execution, package_job, queue_row) in enumerate(
        zip(executions, package_jobs, queue_rows)
    ):
        if not isinstance(execution, Mapping) or not isinstance(
            package_job, Mapping
        ):
            raise ActivationContractError(
                "Activation execution is malformed."
            )
        execution_id = str(execution.get("execution_id"))
        observed_ids.append(execution_id)
        if execution_id != str(plan_ids[index]):
            raise ActivationContractError(
                "Activation execution order drifted."
            )
        job_binding = execution.get("job")
        authorization_binding = execution.get("authorization")
        if not isinstance(job_binding, Mapping) or not isinstance(
            authorization_binding, Mapping
        ):
            raise ActivationContractError(
                "Execution binding is incomplete."
            )
        job_path = _verify_file_binding(
            job_binding,
            base=root,
            label=f"{execution_id} job",
            json_payload=True,
        )
        job = load_json(job_path, label=f"{execution_id} job")
        resources = _validated_job_resources(job)
        if (
            job.get("schema") != JOB_SCHEMA
            or job.get("package_id") != PACKAGE_ID
            or job.get("execution_id") != execution_id
            or job.get("sha256")
            != job_binding.get("canonical_sha256")
            or package_job.get("path")
            != job_path.relative_to(package_dir).as_posix()
            or package_job.get("canonical_sha256")
            != job.get("sha256")
            or package_job.get("sha256") != job_binding.get("sha256")
            or execution.get("resources") != resources
        ):
            raise ActivationContractError(
                "Activation job binding drifted."
            )

        authorization_path = _verify_file_binding(
            authorization_binding,
            base=activation_dir,
            label=f"{execution_id} authorization",
            json_payload=True,
        )
        if authorization_path.relative_to(
            activation_dir
        ).as_posix() != f"authorizations/{execution_id}.json":
            raise ActivationContractError(
                "Execution authorization path drifted."
            )
        authorization = load_json(
            authorization_path,
            label=f"{execution_id} authorization",
        )
        validate_authorization_payload(
            authorization,
            execution=execution,
            manifest=manifest,
        )
        authorizations.append(authorization)
        expected_files.add(authorization_path)
        expected_queue = [
            execution_id,
            job_path.relative_to(root).as_posix(),
            str(job_binding["sha256"]),
            authorization_path.relative_to(root).as_posix(),
            str(authorization_binding["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if queue_row != expected_queue or index != int(
            execution.get("queue_index", -1)
        ):
            raise ActivationContractError(
                "Activation queue binding drifted."
            )

    if len(set(observed_ids)) != DIRECT_EXECUTION_COUNT:
        raise ActivationContractError(
            "Activation execution IDs collide."
        )
    observed_files = {
        path for path in activation_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ActivationContractError(
            "Activation recursive closure drifted."
        )
    if any(path.is_symlink() for path in activation_dir.rglob("*")):
        raise ActivationContractError(
            "Activation contains a symlink."
        )
    return {
        "manifest": manifest,
        "package_manifest": package_manifest,
        "execution_plan": plan,
        "calibration": package["calibration"],
        "package_validation": package["validator_receipt"],
        "authorizations": authorizations,
    }


__all__ = [
    "ACTIVATION_ID",
    "ACTIVATION_RELATIVE",
    "ACTIVATION_SCHEMA",
    "ActivationContractError",
    "AUTHORIZATION_SCHEMA",
    "BATCH_NAME",
    "CALIBRATION_CANONICAL_SHA256",
    "CALIBRATION_FILE_SHA256",
    "CALIBRATION_RECEIPT_NAME",
    "CONTROL_FILES",
    "DIRECT_EXECUTION_COUNT",
    "EXECUTION_PLAN_CANONICAL_SHA256",
    "EXECUTION_PLAN_FILE_SHA256",
    "EXPANDED_DRY_RUN_KIND",
    "FACTORY_DRY_RUN_KIND",
    "JOB_RESOURCE_KEYS",
    "JOB_SCHEMA",
    "LEAVE_IN_QUEUE",
    "MAX_MATERIALIZE",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_CANONICAL_SHA256",
    "PACKAGE_MANIFEST_FILE_SHA256",
    "PACKAGE_RELATIVE",
    "QUEUE_VARIABLES",
    "REMOTE_DRY_RUN_SCHEMA",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "RESOURCE_STATUS",
    "RUNTIME_RELATIVE",
    "SOURCE_ARCHIVE_SHA256",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "expanded_dry_run_submit_text",
    "file_binding",
    "json_binding",
    "load_json",
    "remote_dry_run_contract",
    "repo_root_from_script",
    "sha256_file",
    "validate_activation",
    "validate_authorization_payload",
    "validate_remote_dry_run",
    "validate_remote_dry_run_text",
    "validate_remote_expanded_dry_run",
    "validate_remote_expanded_dry_run_text",
    "validate_remote_factory_dry_run",
    "validate_remote_factory_dry_run_text",
    "validate_sealed_package",
    "validate_provisional_calibration_payload",
    "validate_submit_text",
    "verify_self_digest",
]
