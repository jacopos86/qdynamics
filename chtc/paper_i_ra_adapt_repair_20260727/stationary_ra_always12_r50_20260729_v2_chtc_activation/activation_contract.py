#!/usr/bin/env python3
"""Validation contract for the authorized execution overlay of sealed v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_ra_always12_r50_20260729_v2_chtc"
)
ACTIVATION_ID = (
    "paper_i_ra_adapt_stationary_ra_always12_r50_20260729_v2_"
    "chtc_activation_v1"
)
BATCH_NAME = (
    "paper-i-ra-adapt-stationary-ra-always12-r50-20260729-v2"
)
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_ra_always12_r50_20260729_v2_chtc"
)
ACTIVATION_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_ra_always12_r50_20260729_v2_chtc_activation"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "467be866ac8abd01b109aefea69112aacb1b658da37c66eac92a4976d387fe9f"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "f1c0b0abe107d6a35882ef11f4a649d20c9e08c16719b30e59226ff29e6813fb"
)
EXECUTION_PLAN_CANONICAL_SHA256 = (
    "df3f215c66901fbac868fd7253e6d6f2a5edd97d7d2e43fe5aa9deb41dc5d45a"
)
EXECUTION_PLAN_FILE_SHA256 = (
    "a9679495b5c664877004cad7478b216d5085251bf5a153c24c9560bde6b2963e"
)
SOURCE_ARCHIVE_SHA256 = (
    "1407947832291ab15ad91b0455058a6de689dac42cd1cb5282a76eeafbbc409d"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_always_commutation_reduced_execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_always_commutation_reduced_activation_manifest_v1"
)
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "build_attempt_archive.py",
    "execute_authorized_job.sh",
    "submit.sub",
)


class ActivationContractError(ValueError):
    """Raised when the activation overlay is incomplete or drifts."""


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


def sealed_package_inventory(package_dir: Path) -> tuple[str, ...]:
    if not package_dir.is_dir() or package_dir.is_symlink():
        raise ActivationContractError("Sealed v2 package is unavailable.")
    files = tuple(
        sorted(
            path.relative_to(package_dir).as_posix()
            for path in package_dir.rglob("*")
            if path.is_file()
        )
    )
    if any(path.is_symlink() for path in package_dir.rglob("*")):
        raise ActivationContractError("Sealed v2 contains a symlink.")
    return files


def validate_authorization_payload(
    authorization: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    verify_self_digest(authorization, label="execution authorization")
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "package_id": PACKAGE_ID,
        "activation_id": ACTIVATION_ID,
        "batch_name": BATCH_NAME,
        "execution_id": execution.get("execution_id"),
        "job_sha256": execution.get("job", {}).get("canonical_sha256"),
        "job_file_sha256": execution.get("job", {}).get("sha256"),
        "package_manifest_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_CANONICAL_SHA256,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "activation_control_plane_sha256": manifest.get(
            "activation_control_plane_sha256"
        ),
        "remote_image_path": REMOTE_IMAGE_PATH,
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "remote_image_byte_verification_passed": True,
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ActivationContractError("Execution authorization binding drifted.")
    if not authorization.get("authorization_id") or not authorization.get(
        "authorized_utc"
    ):
        raise ActivationContractError("Execution authorization is undated.")


def _parse_queue(path: Path) -> list[list[str]]:
    rows = [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(len(row) != 9 for row in rows):
        raise ActivationContractError("Activation queue is malformed.")
    return rows


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package_dir = root / PACKAGE_RELATIVE
    activation_dir = root / ACTIVATION_RELATIVE
    if len(sealed_package_inventory(package_dir)) != 25:
        raise ActivationContractError("Sealed v2 recursive closure drifted.")

    package_manifest_path = package_dir / "package_manifest.json"
    package_manifest = load_json(
        package_manifest_path, label="sealed package manifest"
    )
    verify_self_digest(package_manifest, label="sealed package manifest")
    plan_path = package_dir / "execution_plan.json"
    plan = load_json(plan_path, label="sealed execution plan")
    verify_self_digest(plan, label="sealed execution plan")
    if (
        sha256_file(package_manifest_path)
        != PACKAGE_MANIFEST_FILE_SHA256
        or package_manifest.get("sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or sha256_file(plan_path) != EXECUTION_PLAN_FILE_SHA256
        or plan.get("sha256") != EXECUTION_PLAN_CANONICAL_SHA256
        or sha256_file(package_dir / "source_locked.tar.gz")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise ActivationContractError("Sealed v2 authority drifted.")

    manifest_path = activation_dir / "activation_manifest.json"
    manifest = load_json(manifest_path, label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    fixed = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "batch_name": BATCH_NAME,
        "direct_execution_count": 12,
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if any(manifest.get(key) != value for key, value in fixed.items()):
        raise ActivationContractError("Activation state drifted.")
    if "cluster_id" in manifest:
        raise ActivationContractError("Pre-submit activation claims a cluster.")

    package_binding = manifest.get("sealed_package")
    if not isinstance(package_binding, Mapping):
        raise ActivationContractError("Sealed package binding is absent.")
    if (
        package_binding.get("manifest_canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or package_binding.get("manifest_file_sha256")
        != PACKAGE_MANIFEST_FILE_SHA256
        or package_binding.get("execution_plan_canonical_sha256")
        != EXECUTION_PLAN_CANONICAL_SHA256
        or package_binding.get("execution_plan_file_sha256")
        != EXECUTION_PLAN_FILE_SHA256
        or package_binding.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise ActivationContractError("Activation lost sealed-package binding.")

    controls = manifest.get("control_plane")
    if not isinstance(controls, list) or [
        row.get("path") for row in controls if isinstance(row, Mapping)
    ] != list(CONTROL_FILES):
        raise ActivationContractError("Activation control plane drifted.")
    for row in controls:
        if not isinstance(row, Mapping):
            raise ActivationContractError("Control-plane row is malformed.")
        _verify_file_binding(
            row, base=activation_dir, label=f"control {row.get('path')}"
        )
    observed_control_digest = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    if observed_control_digest != manifest.get(
        "activation_control_plane_sha256"
    ):
        raise ActivationContractError("Control-plane digest drifted.")

    queue_binding = manifest.get("queue")
    if not isinstance(queue_binding, Mapping):
        raise ActivationContractError("Activation queue binding is absent.")
    queue_path = _verify_file_binding(
        queue_binding, base=activation_dir, label="activation queue"
    )
    queue_rows = _parse_queue(queue_path)

    executions = manifest.get("executions")
    if not isinstance(executions, list) or len(executions) != 12:
        raise ActivationContractError("Activation execution count drifted.")
    package_jobs = package_manifest.get("jobs")
    if not isinstance(package_jobs, list) or len(package_jobs) != 12:
        raise ActivationContractError("Sealed package job count drifted.")

    authorizations: list[dict[str, Any]] = []
    expected_files = {
        *(activation_dir / name for name in CONTROL_FILES),
        activation_dir / "activation_manifest.json",
        activation_dir / "queue.tsv",
    }
    for index, (execution, package_job, queue_row) in enumerate(
        zip(executions, package_jobs, queue_rows, strict=True)
    ):
        if not isinstance(execution, Mapping) or not isinstance(
            package_job, Mapping
        ):
            raise ActivationContractError("Activation execution is malformed.")
        execution_id = str(execution.get("execution_id"))
        job_binding = execution.get("job")
        authorization_binding = execution.get("authorization")
        if not isinstance(job_binding, Mapping) or not isinstance(
            authorization_binding, Mapping
        ):
            raise ActivationContractError("Execution binding is incomplete.")
        job_path = _verify_file_binding(
            job_binding,
            base=root,
            label=f"{execution_id} job",
            json_payload=True,
        )
        job = load_json(job_path, label=f"{execution_id} job")
        if (
            job.get("execution_id") != execution_id
            or job.get("sha256") != job_binding.get("canonical_sha256")
            or package_job.get("path")
            != job_path.relative_to(package_dir).as_posix()
            or package_job.get("canonical_sha256") != job.get("sha256")
        ):
            raise ActivationContractError("Activation job binding drifted.")
        authorization_path = _verify_file_binding(
            authorization_binding,
            base=activation_dir,
            label=f"{execution_id} authorization",
            json_payload=True,
        )
        authorization = load_json(
            authorization_path, label=f"{execution_id} authorization"
        )
        validate_authorization_payload(
            authorization, execution=execution, manifest=manifest
        )
        authorizations.append(authorization)
        expected_files.add(authorization_path)
        expected_queue = [
            execution_id,
            job_path.relative_to(root).as_posix(),
            str(job_binding["sha256"]),
            authorization_path.relative_to(root).as_posix(),
            str(authorization_binding["sha256"]),
            str(execution["resources"]["request_cpus"]),
            str(execution["resources"]["request_memory_mb"]),
            str(execution["resources"]["request_disk_mb"]),
            str(execution["resources"]["max_runtime_seconds"]),
        ]
        if queue_row != expected_queue or index != int(
            execution.get("queue_index", -1)
        ):
            raise ActivationContractError("Activation queue binding drifted.")

    observed_files = {
        path for path in activation_dir.rglob("*") if path.is_file()
    }
    if observed_files != expected_files:
        raise ActivationContractError("Activation recursive closure drifted.")
    if any(path.is_symlink() for path in activation_dir.rglob("*")):
        raise ActivationContractError("Activation contains a symlink.")
    return {
        "manifest": manifest,
        "package_manifest": package_manifest,
        "execution_plan": plan,
        "authorizations": authorizations,
    }


__all__ = [
    "ACTIVATION_ID",
    "ACTIVATION_RELATIVE",
    "ACTIVATION_SCHEMA",
    "ActivationContractError",
    "AUTHORIZATION_SCHEMA",
    "BATCH_NAME",
    "CONTROL_FILES",
    "EXECUTION_PLAN_CANONICAL_SHA256",
    "EXECUTION_PLAN_FILE_SHA256",
    "PACKAGE_ID",
    "PACKAGE_MANIFEST_CANONICAL_SHA256",
    "PACKAGE_MANIFEST_FILE_SHA256",
    "PACKAGE_RELATIVE",
    "REMOTE_IMAGE_PATH",
    "REMOTE_IMAGE_SHA256",
    "SOURCE_ARCHIVE_SHA256",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "file_binding",
    "json_binding",
    "load_json",
    "repo_root_from_script",
    "sealed_package_inventory",
    "sha256_file",
    "validate_activation",
    "validate_authorization_payload",
    "verify_self_digest",
]
