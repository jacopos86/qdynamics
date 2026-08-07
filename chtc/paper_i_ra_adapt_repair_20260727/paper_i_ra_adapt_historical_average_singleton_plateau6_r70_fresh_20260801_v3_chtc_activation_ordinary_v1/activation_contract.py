#!/usr/bin/env python3
"""Fail-closed contract for the six-row historical-average RA activation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping


sys.dont_write_bytecode = True


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260801_v3_chtc"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_v1"
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_"
    "ordinary_activation_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r70_"
    "execution_authorization_v1"
)
BATCH_NAME = (
    "paper-i-ra-historical-average-singleton-plateau6-r70-fresh-"
    "20260801-v3"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
DIRECT_EXECUTION_COUNT = 6
RUNTIME_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{PACKAGE_ID}_runtime"
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
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "execute_authorized_job.sh",
    "build_attempt_archive.py",
    "submit.sub.in",
)
GENERATED_PATHS = (
    "authorizations",
    "queue.tsv",
    "activation_manifest.json",
    "submit.sub",
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ActivationContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> str:
    expected = payload.get("sha256")
    body = dict(payload)
    body.pop("sha256", None)
    observed = canonical_sha256(body)
    if (
        not isinstance(expected, str)
        or not _HEX64.fullmatch(expected)
        or expected != observed
    ):
        raise ActivationContractError(f"{label} digest drifted.")
    return observed


def repo_root_from_script(script: str | Path) -> Path:
    for parent in Path(script).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / "pipelines").is_dir():
            return parent
    raise ActivationContractError("Repository root could not be resolved.")


def activation_relative() -> PurePosixPath:
    return PurePosixPath("chtc/paper_i_ra_adapt_repair_20260727") / ACTIVATION_ID


def package_relative() -> PurePosixPath:
    return PurePosixPath("chtc/paper_i_ra_adapt_repair_20260727") / PACKAGE_ID


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
    canonical = verify_self_digest(payload, label=path.name)
    result = file_binding(path, relative_to=relative_to)
    result["canonical_sha256"] = canonical
    return result


def _load_package_contract(package_dir: Path) -> ModuleType:
    path = package_dir / "package_contract.py"
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError("Package contract is unavailable.")
    spec = importlib.util.spec_from_file_location(
        "historical_average_ra_r70_sealed_package_contract", path
    )
    if spec is None or spec.loader is None:
        raise ActivationContractError("Package contract cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_package_validator(package_dir: Path) -> dict[str, Any]:
    validator = package_dir / "validate_package.py"
    if not validator.is_file() or validator.is_symlink():
        raise ActivationContractError("Package validator is unavailable.")
    completed = subprocess.run(
        [sys.executable, "-B", str(validator)],
        cwd=package_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ActivationContractError(
            f"Sealed package validator failed: {detail[-1000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ActivationContractError("Package validator returned no receipt.")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ActivationContractError(
            "Package validator receipt is not JSON."
        ) from exc
    if not isinstance(result, dict) or result.get("status") not in {
        "passed",
        "passed_inert_six_rows",
    }:
        raise ActivationContractError("Package validator did not pass.")
    return result


def _bound_package_path(
    package_dir: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    value = binding.get("path")
    if not isinstance(value, str):
        raise ActivationContractError(f"{label} path is absent.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ActivationContractError(f"Unsafe {label} path: {value}")
    path = package_dir.joinpath(*pure.parts)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != binding.get("sha256")
        or path.stat().st_size != binding.get("size_bytes")
    ):
        raise ActivationContractError(f"{label} byte binding drifted.")
    return path


def validate_sealed_package(repo_root: Path) -> dict[str, Any]:
    package_dir = repo_root / package_relative()
    contract = _load_package_contract(package_dir)
    validator_receipt = _run_package_validator(package_dir)
    manifest_path = package_dir / "package_manifest.json"
    plan_path = package_dir / "execution_plan.json"
    audit_path = package_dir / "source_lock_audit.json"
    manifest = load_json(manifest_path, label="package manifest")
    plan = load_json(plan_path, label="execution plan")
    audit = load_json(audit_path, label="source-lock audit")
    verify_self_digest(manifest, label="package manifest")
    verify_self_digest(plan, label="execution plan")
    verify_self_digest(audit, label="source-lock audit")
    expected_ids = list(contract.expected_execution_ids())
    if (
        manifest.get("schema") != contract.PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_six_rows"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != contract.CAMPAIGN_ID
        or manifest.get("row_count") != DIRECT_EXECUTION_COUNT
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("source_horizon") != contract.SOURCE_HORIZON
        or manifest.get("target_horizon") != contract.TARGET_HORIZON
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
        or plan.get("execution_ids") != expected_ids
        or plan.get("row_count") != DIRECT_EXECUTION_COUNT
        or plan.get("ordinary_cluster") is not True
        or plan.get("bounded_factory") is not False
        or plan.get("success_rows_leave_queue") is not False
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("submitted") is not False
        or audit.get("status") != "passed"
        or audit.get("cell_count") != DIRECT_EXECUTION_COUNT
    ):
        raise ActivationContractError("Sealed package authority drifted.")
    source_archive_binding = manifest.get("source_archive")
    if not isinstance(source_archive_binding, Mapping):
        raise ActivationContractError("Source archive binding is absent.")
    source_archive = _bound_package_path(
        package_dir, source_archive_binding, label="source archive"
    )
    if source_archive.relative_to(package_dir).as_posix() != (
        "source/source_locked.tar.gz"
    ):
        raise ActivationContractError("Source archive identity drifted.")
    package_queue = _bound_package_path(
        package_dir,
        manifest.get("queue", {}),
        label="package queue",
    )
    if sha256_file(package_queue) != plan.get("queue_sha256"):
        raise ActivationContractError("Execution plan lost its queue binding.")
    manifest_jobs = manifest.get("jobs")
    if not isinstance(manifest_jobs, list) or len(manifest_jobs) != 6:
        raise ActivationContractError("Package job closure drifted.")
    jobs: list[dict[str, Any]] = []
    for index, (execution_id, row) in enumerate(
        zip(expected_ids, manifest_jobs, strict=True)
    ):
        if not isinstance(row, Mapping) or row.get("execution_id") != execution_id:
            raise ActivationContractError(f"Package job row {index} drifted.")
        job_path = _bound_package_path(
            package_dir, row, label=f"job {execution_id}"
        )
        job = load_json(job_path, label=f"job {execution_id}")
        verify_self_digest(job, label=f"job {execution_id}")
        resources = job.get("resources")
        if (
            job.get("schema") != contract.JOB_SCHEMA
            or job.get("execution_id") != execution_id
            or job.get("package_id") != PACKAGE_ID
            or job.get("run_class") != "candidate"
            or job.get("execution_mode") != "fresh_0_to_70"
            or job.get("target_horizon") != 70
            or job.get("candidate_representation") != "single_pauli_word_v1"
            or job.get("insertion_policy") != "plateau_commutation"
            or job.get("execution_authorized") is not False
            or job.get("submission_authorized") is not False
            or job.get("submitted") is not False
            or not isinstance(resources, Mapping)
        ):
            raise ActivationContractError(f"Job semantics drifted: {execution_id}")
        values = tuple(
            resources.get(key)
            for key in (
                "request_cpus",
                "request_memory_mb",
                "request_disk_mb",
                "max_runtime_seconds",
            )
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ActivationContractError(f"Job resources drifted: {execution_id}")
        jobs.append(
            {
                "execution_id": execution_id,
                "path": job_path,
                "payload": job,
                "resources": resources,
            }
        )
    return {
        "contract": contract,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "plan": plan,
        "plan_path": plan_path,
        "audit": audit,
        "audit_path": audit_path,
        "source_archive": source_archive,
        "source_archive_binding": dict(source_archive_binding),
        "jobs": jobs,
        "validator_receipt": validator_receipt,
    }


def render_submit(*, source_archive_sha256: str) -> str:
    if not _HEX64.fullmatch(source_archive_sha256):
        raise ActivationContractError("Source archive digest is invalid.")
    activation_rel = activation_relative().as_posix()
    package_rel = package_relative().as_posix()
    template_path = Path(__file__).resolve().parent / "submit.sub.in"
    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "@@ACTIVATION_REL@@": activation_rel,
        "@@PACKAGE_REL@@": package_rel,
        "@@SOURCE_ARCHIVE_SHA256@@": source_archive_sha256,
        "@@IMAGE_PATH@@": IMAGE_PATH,
        "@@IMAGE_SHA256@@": IMAGE_SHA256,
        "@@BATCH_NAME@@": BATCH_NAME,
        "@@RUNTIME_REL@@": RUNTIME_RELATIVE,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if "@@" in text:
        raise ActivationContractError("Submit template has unresolved fields.")
    validate_submit_text(text)
    return text


def validate_submit_text(text: str) -> None:
    required = (
        'when_to_transfer_output = ON_EXIT_OR_EVICT',
        f'+JobBatchName = "{BATCH_NAME}"',
        '+HolsteinLifecycleMode = "ordinary_unheld_v1"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "periodic_release = False",
        "leave_in_queue = False",
        "stream_output = False",
        "stream_error = False",
        "queue " + ",".join(QUEUE_VARIABLES) + " from ",
    )
    if any(token not in text for token in required):
        raise ActivationContractError("Submit descriptor contract drifted.")
    lowered = text.lower()
    if "max_materialize" in lowered or "max_idle" in lowered:
        raise ActivationContractError("Ordinary activation cannot be a factory.")
    if re.search(r"(?im)^\s*hold\s*=", text):
        raise ActivationContractError("Ordinary activation cannot submit held.")
    if len(re.findall(r"(?im)^\s*leave_in_queue\s*=", text)) != 1:
        raise ActivationContractError("Lifecycle assignment must be unique.")
    if re.search(r"(?im)^\s*periodic_release\s*=\s*true\s*$", text):
        raise ActivationContractError("Automatic release is forbidden.")

def authorization_payload(
    *,
    package: Mapping[str, Any],
    job_row: Mapping[str, Any],
    control_sha256: str,
    authorized_utc: str,
) -> dict[str, Any]:
    """Return the exact single-cell authority consumed by ``run_cell.py``."""
    if not _HEX64.fullmatch(control_sha256):
        raise ActivationContractError("Control-plane digest is invalid.")
    if not authorized_utc.strip():
        raise ActivationContractError("Authorization timestamp is absent.")
    job = job_row["payload"]
    execution_id = job_row["execution_id"]
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "passed",
        "authorization_id": f"{ACTIVATION_ID}__{execution_id}",
        "authorization_source": "explicit_user_request_2026-08-01",
        "authorized_utc": authorized_utc,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "campaign_id": package["manifest"]["campaign_id"],
        "execution_id": execution_id,
        "job_spec_sha256": job["sha256"],
        "package_manifest_sha256": package["manifest"]["sha256"],
        "protocol_sha256": job["protocol_sha256"],
        "protocol_file_sha256": job["protocol_file_sha256"],
        "source_archive_sha256": package["source_archive_binding"]["sha256"],
        "scope": "single_cell_chtc_execution_only",
        "authorization_kind": (
            "explicit_user_execution_and_submission_authority"
        ),
        "activation_control_plane_sha256": control_sha256,
        "remote_image_path": IMAGE_PATH,
        "remote_image_sha256": IMAGE_SHA256,
        "remote_image_byte_verification_required_before_submit": True,
        "remote_image_byte_verification_passed": False,
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_pending_remote_preflight",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adoption_authorized": False,
    }


def authorization_relative(execution_id: str) -> PurePosixPath:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", execution_id):
        raise ActivationContractError("Unsafe execution id for authorization.")
    return PurePosixPath("authorizations") / f"{execution_id}.json"


def _control_bindings(activation_dir: Path) -> list[dict[str, Any]]:
    return [
        file_binding(activation_dir / name, relative_to=activation_dir)
        for name in CONTROL_FILES
    ]


def _parse_queue(path: Path) -> list[list[str]]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError("Activation queue is unsafe.")
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    if len(rows) != 6 or any(len(row) != len(QUEUE_VARIABLES) for row in rows):
        raise ActivationContractError("Activation queue shape drifted.")
    return rows


def _sealed_package_binding(package: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "path": package_relative().as_posix(),
        "manifest": json_binding(package["manifest_path"], relative_to=repo_root),
        "execution_plan": json_binding(package["plan_path"], relative_to=repo_root),
        "source_lock_audit": json_binding(package["audit_path"], relative_to=repo_root),
        "source_archive": file_binding(package["source_archive"], relative_to=repo_root),
        "validator_status": package["validator_receipt"]["status"],
    }


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = validate_sealed_package(root)
    activation_dir = root / activation_relative()
    if Path(__file__).resolve().parent != activation_dir:
        raise ActivationContractError("Activation directory drifted.")
    controls = _control_bindings(activation_dir)
    control_sha = canonical_sha256(controls)
    submit_path = activation_dir / "submit.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    expected_submit = render_submit(
        source_archive_sha256=package["source_archive_binding"]["sha256"]
    )
    if submit_text != expected_submit:
        raise ActivationContractError("Rendered submit descriptor drifted.")

    expected_ids = [row["execution_id"] for row in package["jobs"]]
    authorizations: list[dict[str, Any]] = []
    authorization_paths: dict[str, Path] = {}
    authorized_utc: str | None = None
    for job_row in package["jobs"]:
        execution_id = job_row["execution_id"]
        authorization_path = activation_dir.joinpath(
            *authorization_relative(execution_id).parts
        )
        authorization = load_json(
            authorization_path,
            label=f"execution authorization {execution_id}",
        )
        verify_self_digest(
            authorization,
            label=f"execution authorization {execution_id}",
        )
        row_authorized_utc = authorization.get("authorized_utc")
        if not isinstance(row_authorized_utc, str) or not row_authorized_utc:
            raise ActivationContractError("Authorization timestamp is absent.")
        expected_authorization = authorization_payload(
            package=package,
            job_row=job_row,
            control_sha256=control_sha,
            authorized_utc=row_authorized_utc,
        )
        if set(authorization) != set(expected_authorization) | {"sha256"}:
            raise ActivationContractError(
                f"Authorization field closure drifted: {execution_id}."
            )
        for key, value in expected_authorization.items():
            if authorization.get(key) != value:
                raise ActivationContractError(
                    f"Authorization drifted at {execution_id}:{key}."
                )
        if authorized_utc is None:
            authorized_utc = row_authorized_utc
        elif authorized_utc != row_authorized_utc:
            raise ActivationContractError("Authorization timestamps diverged.")
        authorization_paths[execution_id] = authorization_path
        authorizations.append(
            {
                "execution_id": execution_id,
                **json_binding(authorization_path, relative_to=activation_dir),
            }
        )
    assert authorized_utc is not None

    queue_path = activation_dir / "queue.tsv"
    queue_rows = _parse_queue(queue_path)
    expected_executions: list[dict[str, Any]] = []
    for index, (queue_row, job_row) in enumerate(
        zip(queue_rows, package["jobs"], strict=True)
    ):
        job = job_row["payload"]
        resources = job_row["resources"]
        authorization_path = authorization_paths[job_row["execution_id"]]
        auth_binding = file_binding(authorization_path, relative_to=root)
        expected = [
            job_row["execution_id"],
            job_row["path"].relative_to(root).as_posix(),
            sha256_file(job_row["path"]),
            authorization_path.relative_to(root).as_posix(),
            auth_binding["sha256"],
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if queue_row != expected:
            raise ActivationContractError(f"Queue row {index} drifted.")
        expected_executions.append(
            {
                "queue_index": index,
                "execution_id": job_row["execution_id"],
                "job": json_binding(job_row["path"], relative_to=root),
                "resources": {
                    key: resources[key]
                    for key in (
                        "request_cpus",
                        "request_memory_mb",
                        "request_disk_mb",
                        "max_runtime_seconds",
                    )
                },
            }
        )

    manifest_path = activation_dir / "activation_manifest.json"
    manifest = load_json(manifest_path, label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    fixed_manifest = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "campaign_id": package["manifest"]["campaign_id"],
        "batch_name": BATCH_NAME,
        "run_class": "candidate",
        "execution_target": "chtc",
        "direct_execution_count": 6,
        "queue_variables": list(QUEUE_VARIABLES),
        "activation_control_plane_sha256": control_sha,
        "operational_mode": "ordinary_unheld_v1",
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_pending_remote_preflight",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adopted": False,
    }
    for key, value in fixed_manifest.items():
        if manifest.get(key) != value:
            raise ActivationContractError(f"Activation manifest drifted at {key}.")
    expected_keys = set(fixed_manifest) | {
        "authorized_utc",
        "sealed_package",
        "remote_image",
        "control_plane",
        "execution_authorizations",
        "executions",
        "queue",
        "submit_descriptor",
        "pre_submit_requirements",
        "sha256",
    }
    if set(manifest) != expected_keys:
        raise ActivationContractError("Activation manifest field closure drifted.")
    if manifest.get("authorized_utc") != authorized_utc:
        raise ActivationContractError("Activation timestamp drifted.")
    if manifest.get("sealed_package") != _sealed_package_binding(package, root):
        raise ActivationContractError("Sealed package binding drifted.")
    if manifest.get("remote_image") != {
        "path": IMAGE_PATH,
        "sha256": IMAGE_SHA256,
        "byte_verification_required_before_submit": True,
        "byte_verification_passed": False,
    }:
        raise ActivationContractError("Remote image lock drifted.")
    if manifest.get("control_plane") != controls:
        raise ActivationContractError("Control-plane binding drifted.")
    if manifest.get("execution_authorizations") != authorizations:
        raise ActivationContractError("Authorization bindings drifted.")
    if manifest.get("executions") != expected_executions:
        raise ActivationContractError("Execution bindings drifted.")
    if manifest.get("queue") != file_binding(queue_path, relative_to=activation_dir):
        raise ActivationContractError("Queue binding drifted.")
    if manifest.get("submit_descriptor") != file_binding(
        submit_path, relative_to=activation_dir
    ):
        raise ActivationContractError("Submit binding drifted.")
    if manifest.get("pre_submit_requirements") != [
        "verify_remote_image_exact_path_size_and_sha256",
        "validate_exact_submit_descriptor_lifecycle",
        "condor_submit_dry_run_exact_descriptor",
        "check_exact_batch_and_execution_id_collisions",
        "check_home_quota_available",
    ]:
        raise ActivationContractError("Pre-submit requirements drifted.")
    actual_files = {
        path.relative_to(activation_dir).as_posix()
        for path in activation_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    expected_generated_files = {
        "queue.tsv",
        "activation_manifest.json",
        "submit.sub",
        *(authorization_relative(execution_id).as_posix() for execution_id in expected_ids),
    }
    if actual_files != set(CONTROL_FILES) | expected_generated_files:
        raise ActivationContractError("Activation file closure drifted.")
    return {
        "status": "passed",
        "activation_id": ACTIVATION_ID,
        "activation_manifest_sha256": manifest["sha256"],
        "batch_name": BATCH_NAME,
        "direct_execution_count": 6,
        "submission_state": "authorized_pending_remote_preflight",
        "ordinary_held": False,
        "factory": False,
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "ActivationContractError",
    "activation_relative",
    "authorization_payload",
    "authorization_relative",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "file_binding",
    "json_binding",
    "load_json",
    "package_relative",
    "render_submit",
    "repo_root_from_script",
    "sha256_file",
    "validate_activation",
    "validate_sealed_package",
    "validate_submit_text",
    "verify_self_digest",
]
