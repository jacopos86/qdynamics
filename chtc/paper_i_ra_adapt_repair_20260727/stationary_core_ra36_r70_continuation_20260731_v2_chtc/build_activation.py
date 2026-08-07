#!/usr/bin/env python3
"""Finalize the inert r70 scaffold as a new atomic ordinary-held v3 sibling.

This program never edits v2.  It first authenticates the nine external
controlled-cycle predecessor bindings, every resume/source/runtime/image file,
four scheduler-backed resource observations, and 36 external execution
authorizations.  Only after all relations close does it stage and atomically
rename a new v3 sibling containing deterministic jobs and a held submit file.
It never contacts CHTC and never submits or releases a job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from scaffold_contract import (  # noqa: E402
    ACTIVATION_AUTHORIZATION_SCHEMA,
    ACTIVATION_INPUT_SCHEMA,
    CELL_COUNT,
    IMAGE_VERIFICATION_SCHEMA,
    PACKAGE_ID,
    PACKAGE_RELATIVE_ROOT,
    PENDING_PREDECESSORS,
    RESOURCE_EVIDENCE_SCHEMA,
    RESOURCE_OBSERVATION_SCHEMA,
    RUNTIME_BUNDLE_MANIFEST_SCHEMA,
    SCAFFOLD_MANIFEST_NAME,
    SCHEDULER_TERMINAL_RECEIPT_SCHEMA,
    SHA256_RE,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TRANSFER_PLAN_NAME,
    ScaffoldContractError,
    _bound_path,
    _load_bound_controlled_json,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    load_json,
    repo_file_binding,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    transfer_path_is_regular_file,
    validate_predecessor_binding,
    validate_resume_input_contents,
    verify_exact_binding,
    verify_self_digest,
)
from validate_scaffold import validate_scaffold  # noqa: E402


FINAL_PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_ra36_r70_"
    "continuation_20260731_v3_chtc"
)
FINAL_CAMPAIGN_ID = (
    "paper_i_ra_adapt_stationary_core_r70_continuation_v3"
)
FINAL_PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_ra36_r70_continuation_20260731_v3_chtc"
)
FINAL_JOB_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_authorized_job_v3"
)
FINAL_CONTROL_PLANE_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_activation_control_plane_v3"
)
FINAL_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_activation_manifest_v3"
)
FINAL_BATCH_NAME = (
    "paper-i-ra-adapt-stationary-core36-r70-resume-"
    "20260731-v3-ordinary-held-v1"
)
LIFECYCLE_MODE = "ordinary_held_exact_proc_release_v1"
ACTIVATION_CONTROL_PLANE_NAME = "activation_control_plane.json"
ACTIVATION_INPUT_COPY_NAME = "activation_inputs.json"
ACTIVATION_MANIFEST_NAME = "activation_manifest.json"
ACTIVATION_QUEUE_NAME = "queue.tsv"
SUBMIT_NAME = "submit.sub"
FINAL_JOBS_DIR = "jobs"
FINAL_AUTHORIZATIONS_DIR = "authorizations"
FINAL_PREDECESSOR_BINDINGS_DIR = "predecessor_bindings"
QUEUE_VARIABLES = (
    "execution_id",
    "job_path",
    "authorization_path",
    "row_bootstrap",
    "runtime_bundle",
    "source_archive",
    "source_manifest",
    "source_delta",
    "resume_archive",
    "image_path",
    "cpus",
    "memory_mb",
    "disk_mb",
    "max_runtime_seconds",
)
RESOURCE_KEYS = (
    "request_cpus",
    "request_memory_mb",
    "request_disk_mb",
    "max_runtime_seconds",
)


@dataclass(frozen=True)
class FinalizationContext:
    """Fully named inert input surface used by the finalizer core."""

    repo_root: Path
    package_dir: Path
    package_relative_root: str
    scaffold_manifest: Mapping[str, Any]
    scaffold_manifest_binding: Mapping[str, Any]
    jobs: Mapping[str, Mapping[str, Any]]
    job_bindings: Mapping[str, Mapping[str, Any]]
    transfer: Mapping[str, Any]
    requirements: Mapping[str, Mapping[str, Any]]
    expected_pending_ids: frozenset[str]
    expected_cell_count: int
    inert_inventory_sha256: str


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScaffoldContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScaffoldContractError(f"{label} must be a list.")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScaffoldContractError(f"{label} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScaffoldContractError(f"{label} must be a non-negative integer.")
    return value


def _utc(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ScaffoldContractError(f"{label} must be an RFC-3339 UTC time.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ScaffoldContractError(f"{label} is not RFC-3339.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ScaffoldContractError(f"{label} is not UTC.")
    return value


def _plain_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "path": str(binding.get("path", "")),
        "sha256": str(binding.get("sha256", "")),
        "size_bytes": binding.get("size_bytes"),
    }
    if binding.get("canonical_sha256") is not None:
        result["canonical_sha256"] = str(binding["canonical_sha256"])
    return result


def _payload_binding(
    *, relative_path: str, payload: bytes, canonical_digest: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if canonical_digest is not None:
        result["canonical_sha256"] = canonical_digest
    return result


def _json_payload(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _repo_json_binding(
    path: Path, *, repo_root: Path, label: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    file_binding = repo_file_binding(path, repo_root=repo_root)
    file_binding["path"] = relative
    _path, parsed = _load_bound_controlled_json(
        binding={**file_binding, "canonical_sha256": load_json(path, label=label).get("sha256")},
        repo_root=repo_root,
        label=label,
    )
    raw = path.read_bytes()
    file_binding["canonical_sha256"] = parsed["sha256"]
    return file_binding, raw, parsed


def _file_inventory_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ScaffoldContractError(f"Unsafe inert package directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ScaffoldContractError(f"Inert package contains a symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return canonical_sha256(rows)


def _production_context() -> FinalizationContext:
    repo_root = repo_root_from_script(__file__)
    package_dir = repo_root / PACKAGE_RELATIVE_ROOT
    result = validate_scaffold(rehash_existing_resumes=True)
    if (
        result.get("status")
        != "passed_inert_scaffold_missing_9_predecessors"
        or result.get("missing_authenticated_resume_count") != len(PENDING_PREDECESSORS)
    ):
        raise ScaffoldContractError("The v2 input is not the exact inert 27+9 scaffold.")
    manifest_path = package_dir / SCAFFOLD_MANIFEST_NAME
    manifest = load_json(manifest_path, label="v2 scaffold manifest")
    verify_self_digest(manifest, label="v2 scaffold manifest")
    manifest_binding = repo_file_binding(manifest_path, repo_root=repo_root)
    manifest_binding["canonical_sha256"] = manifest["sha256"]

    jobs: dict[str, Mapping[str, Any]] = {}
    job_bindings: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(manifest.get("jobs"), label="v2 job bindings"):
        binding = _mapping(raw, label="v2 job binding")
        execution_id = str(binding.get("execution_id", ""))
        path = package_dir / safe_relative_path(
            binding.get("path"), label=f"{execution_id} v2 job path"
        )
        verify_exact_binding(path, binding, label=f"{execution_id} v2 job")
        job = load_json(path, label=f"{execution_id} v2 job")
        verify_self_digest(job, label=f"{execution_id} v2 job")
        jobs[execution_id] = job
        job_bindings[execution_id] = {
            **_plain_binding(binding),
            "path": path.relative_to(repo_root).as_posix(),
        }
    transfer = load_json(package_dir / TRANSFER_PLAN_NAME, label="v2 transfer plan")
    verify_self_digest(transfer, label="v2 transfer plan")
    inventory = load_json(
        package_dir / "predecessor_requirements.json",
        label="v2 predecessor requirements",
    )
    verify_self_digest(inventory, label="v2 predecessor requirements")
    requirements = {
        str(row["execution_id"]): row
        for row in _sequence(inventory.get("requirements"), label="requirements")
        if isinstance(row, Mapping)
    }
    if set(jobs) != set(result_ids := [str(row["execution_id"]) for row in manifest["jobs"]]):
        raise ScaffoldContractError("V2 job identities are duplicated.")
    if len(result_ids) != CELL_COUNT or set(requirements) != set(PENDING_PREDECESSORS):
        raise ScaffoldContractError("V2 finalization identity closure drifted.")
    return FinalizationContext(
        repo_root=repo_root,
        package_dir=package_dir,
        package_relative_root=PACKAGE_RELATIVE_ROOT,
        scaffold_manifest=manifest,
        scaffold_manifest_binding=manifest_binding,
        jobs=jobs,
        job_bindings=job_bindings,
        transfer=transfer,
        requirements=requirements,
        expected_pending_ids=frozenset(PENDING_PREDECESSORS),
        expected_cell_count=CELL_COUNT,
        inert_inventory_sha256=_file_inventory_digest(package_dir),
    )


def _validate_exact_file_path(value: Any, *, label: str) -> str:
    path = safe_relative_path(value, label=label).as_posix()
    if not transfer_path_is_regular_file(path):
        raise ScaffoldContractError(f"{label} is not an exact regular file.")
    return path


def _load_activation_inputs(
    path: Path,
    *,
    context: FinalizationContext,
    require_authorizations: bool,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    binding, raw, payload = _repo_json_binding(
        path, repo_root=context.repo_root, label="activation inputs"
    )
    expected_status = (
        "evidence_and_authorizations_complete"
        if require_authorizations
        else "evidence_complete_authorizations_pending"
    )
    if (
        payload.get("schema") != ACTIVATION_INPUT_SCHEMA
        or payload.get("inert_package_id") != PACKAGE_ID
        or payload.get("final_package_id") != FINAL_PACKAGE_ID
        or payload.get("status") != expected_status
        or payload.get("execution_authorized") is not require_authorizations
        or payload.get("submission_authorized") is not require_authorizations
        or payload.get("submitted") is not False
    ):
        raise ScaffoldContractError("Activation-input header/state drifted.")
    return binding, raw, payload


def _safe_runtime_member(value: Any, *, label: str) -> str:
    return safe_relative_path(value, label=label).as_posix()


def _validate_runtime(
    *, inputs: Mapping[str, Any], context: FinalizationContext
) -> dict[str, Any]:
    bundle = _plain_binding(_mapping(inputs.get("runtime_bundle"), label="runtime bundle"))
    bundle_path = _bound_path(
        binding=bundle, repo_root=context.repo_root, label="runtime bundle"
    )
    bootstrap = _plain_binding(_mapping(inputs.get("row_bootstrap"), label="row bootstrap"))
    bootstrap_path = _bound_path(
        binding=bootstrap, repo_root=context.repo_root, label="row bootstrap"
    )
    if bootstrap_path.stat().st_mode & 0o111 == 0:
        raise ScaffoldContractError("Row bootstrap is not executable.")
    manifest_binding = _mapping(
        inputs.get("runtime_bundle_manifest"), label="runtime manifest binding"
    )
    _manifest_path, manifest = _load_bound_controlled_json(
        binding=manifest_binding,
        repo_root=context.repo_root,
        label="runtime bundle manifest",
    )
    members = _sequence(manifest.get("members"), label="runtime members")
    expected: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(members):
        row = _mapping(raw, label=f"runtime member {index}")
        name = _safe_runtime_member(row.get("path"), label="runtime member path")
        if name in expected:
            raise ScaffoldContractError("Runtime manifest duplicates a member.")
        _positive_int(row.get("size_bytes"), label=f"{name} size")
        _nonnegative_int(row.get("mode"), label=f"{name} mode")
        expected[name] = row
    entrypoint = _safe_runtime_member(
        manifest.get("entrypoint_member"), label="runtime entrypoint"
    )
    entry = _mapping(expected.get(entrypoint), label="runtime entrypoint row")
    if (
        manifest.get("schema") != RUNTIME_BUNDLE_MANIFEST_SCHEMA
        or manifest.get("package_id") != FINAL_PACKAGE_ID
        or manifest.get("status") != "passed"
        or _plain_binding(_mapping(manifest.get("archive"), label="runtime archive"))
        != bundle
        or manifest.get("member_count") != len(expected)
        or not expected
        or entry.get("role") != "row_bootstrap"
        or entry.get("sha256") != bootstrap.get("sha256")
        or entry.get("size_bytes") != bootstrap.get("size_bytes")
        or int(entry.get("mode", 0)) & 0o111 == 0
    ):
        raise ScaffoldContractError("Runtime manifest relation closure failed.")
    observed: set[str] = set()
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            for member in archive:
                name = _safe_runtime_member(member.name, label="runtime tar member")
                row = expected.get(name)
                if (
                    row is None
                    or name in observed
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != row.get("size_bytes")
                    or member.mode & 0o777 != int(row.get("mode", -1)) & 0o777
                ):
                    raise ScaffoldContractError(f"Unsafe runtime member: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ScaffoldContractError(f"Unreadable runtime member: {name}")
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                if size != member.size or digest.hexdigest() != row.get("sha256"):
                    raise ScaffoldContractError(f"Runtime member drifted: {name}")
                observed.add(name)
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ScaffoldContractError("Runtime bundle is not a complete safe tar.gz.") from exc
    if observed != set(expected):
        raise ScaffoldContractError("Runtime tar member closure is incomplete.")
    return {
        "bundle": bundle,
        "manifest": _plain_binding(manifest_binding),
        "manifest_payload": manifest,
        "bootstrap": bootstrap,
    }


def _validate_image(
    *, inputs: Mapping[str, Any], context: FinalizationContext
) -> dict[str, Any]:
    declared = _mapping(inputs.get("image"), label="remote image")
    image = {
        "host": str(declared.get("host", "")),
        "remote_root": str(declared.get("remote_root", "")),
        "path": _validate_exact_file_path(
            declared.get("path"), label="remote image path"
        ),
        "sha256": str(declared.get("sha256", "")),
        "size_bytes": declared.get("size_bytes"),
    }
    if (
        image["host"] != "ap2001.chtc.wisc.edu"
        or image["remote_root"]
        != "/home/jsstrobel/Holstein_phase3_optuna_chtc"
        or not image["path"].endswith(".sif")
        or SHA256_RE.fullmatch(image["sha256"]) is None
        or isinstance(image["size_bytes"], bool)
        or not isinstance(image["size_bytes"], int)
        or image["size_bytes"] < 1
    ):
        raise ScaffoldContractError("Remote image identity is malformed.")
    receipt_binding = _mapping(
        inputs.get("image_verification"), label="image verification binding"
    )
    _path, receipt = _load_bound_controlled_json(
        binding=receipt_binding,
        repo_root=context.repo_root,
        label="image verification receipt",
    )
    remote = _mapping(receipt.get("remote_image"), label="remote image")
    verification = _mapping(receipt.get("verification"), label="image verification")
    receipt_remote = {
        "host": remote.get("host"),
        "remote_root": remote.get("remote_root"),
        "path": remote.get("path"),
        "sha256": remote.get("sha256"),
        "size_bytes": remote.get("size_bytes"),
    }
    if (
        receipt.get("schema") != IMAGE_VERIFICATION_SCHEMA
        or receipt.get("package_id") != FINAL_PACKAGE_ID
        or receipt.get("status") != "passed"
        or receipt_remote != image
        or verification
        != {
            "remote_regular_file": True,
            "remote_sha256_verified": True,
            "remote_size_verified": True,
        }
    ):
        raise ScaffoldContractError("Image verification relation closure failed.")
    _utc(remote.get("verified_utc"), label="remote image verified_utc")
    return {
        "image": image,
        "verification": _plain_binding(receipt_binding),
        "receipt": receipt,
    }


def _resource_bucket(job: Mapping[str, Any]) -> str:
    representation = str(job.get("candidate_representation", ""))
    nph = _positive_int(job.get("nph"), label="job nph")
    if representation not in {"macro_generator_v1", "single_pauli_word_v1"}:
        raise ScaffoldContractError("Unknown resource candidate representation.")
    if nph not in {3, 7}:
        raise ScaffoldContractError("Unknown resource cutoff bucket.")
    return f"{representation}:nph{nph}"


def _resource_values(
    value: Any, *, label: str, exact_keys: bool = True
) -> dict[str, int]:
    row = _mapping(value, label=label)
    if exact_keys and set(row) != set(RESOURCE_KEYS):
        raise ScaffoldContractError(f"{label} field closure drifted.")
    if not set(RESOURCE_KEYS).issubset(row):
        raise ScaffoldContractError(f"{label} lacks a required resource field.")
    return {key: _positive_int(row.get(key), label=f"{label} {key}") for key in RESOURCE_KEYS}


def _validate_resources(
    *, inputs: Mapping[str, Any], context: FinalizationContext
) -> dict[str, Any]:
    evidence_binding = _mapping(
        inputs.get("resource_evidence"), label="resource evidence binding"
    )
    _path, evidence = _load_bound_controlled_json(
        binding=evidence_binding,
        repo_root=context.repo_root,
        label="resource evidence",
    )
    observations = _mapping(evidence.get("observations"), label="resource observations")
    pilot = _mapping(
        evidence.get("r70_worst_bucket_pilot"),
        label="r70 worst-bucket pilot contract",
    )
    expected_buckets = {_resource_bucket(job) for job in context.jobs.values()}
    pilot_execution_id = str(pilot.get("execution_id", ""))
    pilot_job = context.jobs.get(pilot_execution_id)
    if (
        evidence.get("schema") != RESOURCE_EVIDENCE_SCHEMA
        or evidence.get("package_id") != FINAL_PACKAGE_ID
        or evidence.get("status")
        != "passed_for_held_submission_r70_pilot_pending"
        or evidence.get("policy")
        != "r50_history_plus_conservative_r70_headroom_v1"
        or set(observations) != expected_buckets
        or expected_buckets
        != {
            "macro_generator_v1:nph3",
            "macro_generator_v1:nph7",
            "single_pauli_word_v1:nph3",
            "single_pauli_word_v1:nph7",
        }
        or pilot.get("bucket_id") != "single_pauli_word_v1:nph7"
        or pilot.get("status") != "planned_not_executed"
        or pilot.get("initial_state") != "held"
        or pilot.get("broad_release_authorized") is not False
        or pilot.get("pilot_release_requires_separate_exact_proc_authorization")
        is not True
        or not isinstance(pilot_job, Mapping)
        or _resource_bucket(pilot_job) != "single_pauli_word_v1:nph7"
    ):
        raise ScaffoldContractError("Resource-evidence bucket closure failed.")
    result: dict[str, Any] = {}
    for bucket in sorted(expected_buckets):
        binding = _mapping(observations[bucket], label=f"{bucket} observation binding")
        _receipt_path, receipt = _load_bound_controlled_json(
            binding=binding,
            repo_root=context.repo_root,
            label=f"{bucket} resource observation",
        )
        source_job_binding = _mapping(
            receipt.get("source_job"), label=f"{bucket} source-job binding"
        )
        _source_job_path, source_job = _load_bound_controlled_json(
            binding=source_job_binding,
            repo_root=context.repo_root,
            label=f"{bucket} source job",
        )
        terminal_binding = _mapping(
            receipt.get("scheduler_terminal_receipt"),
            label=f"{bucket} scheduler-terminal binding",
        )
        _terminal_path, terminal = _load_bound_controlled_json(
            binding=terminal_binding,
            repo_root=context.repo_root,
            label=f"{bucket} scheduler-terminal receipt",
        )
        observed = _mapping(receipt.get("observed"), label=f"{bucket} observed")
        requested = _resource_values(receipt.get("requested"), label=f"{bucket} requested")
        approved = _resource_values(
            receipt.get("approved_envelope"), label=f"{bucket} approved envelope"
        )
        bucket_jobs = [job for job in context.jobs.values() if _resource_bucket(job) == bucket]
        target_execution_id = str(receipt.get("target_execution_id", ""))
        target_jobs = [
            job for job in bucket_jobs if job.get("execution_id") == target_execution_id
        ]
        source_execution_id = str(receipt.get("source_execution_id", ""))
        source_protocol = _mapping(
            source_job.get("protocol"), label=f"{bucket} source protocol"
        )
        source_resources = _resource_values(
            source_job.get("resources"),
            label=f"{bucket} source resources",
            exact_keys=False,
        )
        peak_memory = _nonnegative_int(observed.get("peak_memory_mb"), label=f"{bucket} peak memory")
        peak_disk = _nonnegative_int(observed.get("peak_disk_mb"), label=f"{bucket} peak disk")
        wall = _nonnegative_int(observed.get("wall_seconds"), label=f"{bucket} wall")
        _nonnegative_int(observed.get("output_archive_bytes"), label=f"{bucket} output bytes")
        if (
            receipt.get("schema") != RESOURCE_OBSERVATION_SCHEMA
            or receipt.get("package_id") != FINAL_PACKAGE_ID
            or receipt.get("bucket_id") != bucket
            or receipt.get("status") != "passed"
            or receipt.get("approval_policy")
            != "r50_history_plus_conservative_r70_headroom_v1"
            or receipt.get("horizon") != SOURCE_HORIZON
            or receipt.get("evidence_role")
            != "r50_history_envelope_basis"
            or len(target_jobs) != 1
            or receipt.get("target_scientific_settings_sha256")
            != target_jobs[0].get("scientific_settings_sha256")
            or source_job.get("execution_id") != source_execution_id
            or source_job.get("horizon") != SOURCE_HORIZON
            or _resource_bucket(source_job) != bucket
            or requested != source_resources
            or receipt.get("source_protocol_sha256")
            != source_protocol.get("sha256")
            or terminal.get("schema") != SCHEDULER_TERMINAL_RECEIPT_SCHEMA
            or terminal.get("status") != "passed"
            or terminal.get("execution_id") != source_execution_id
            or terminal.get("source") != "condor_history_exact_cluster_proc"
            or terminal.get("job_status") != 4
            or terminal.get("exit_code") != 0
            or isinstance(terminal.get("cluster_id"), bool)
            or not isinstance(terminal.get("cluster_id"), int)
            or isinstance(terminal.get("proc_id"), bool)
            or not isinstance(terminal.get("proc_id"), int)
            or isinstance(terminal.get("num_job_starts"), bool)
            or not isinstance(terminal.get("num_job_starts"), int)
            or terminal.get("num_job_starts", 0) < 1
            or isinstance(terminal.get("completion_epoch"), bool)
            or not isinstance(terminal.get("completion_epoch"), int)
            or terminal.get("completion_epoch", 0) < 1
            or peak_memory > requested["request_memory_mb"]
            or peak_disk > requested["request_disk_mb"]
            or wall > requested["max_runtime_seconds"]
            or any(requested[key] > approved[key] for key in RESOURCE_KEYS)
        ):
            raise ScaffoldContractError(f"{bucket} resource observation failed closure.")
        for job in bucket_jobs:
            prior = _resource_values(
                job.get("resources"),
                label=f"{bucket} v2 resources",
                exact_keys=False,
            )
            if any(prior[key] > approved[key] for key in RESOURCE_KEYS):
                raise ScaffoldContractError(f"{bucket} approved envelope shrinks a v2 request.")
        result[bucket] = {
            "binding": _plain_binding(binding),
            "receipt": receipt,
            "approved_envelope": approved,
        }
    return {
        "binding": _plain_binding(evidence_binding),
        "evidence": evidence,
        "buckets": result,
        "r70_worst_bucket_pilot": dict(pilot),
    }


def _transfer_roles(context: FinalizationContext) -> dict[str, dict[str, Mapping[str, Any]]]:
    rows = _sequence(context.transfer.get("rows"), label="transfer rows")
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    for raw in rows:
        row = _mapping(raw, label="transfer row")
        execution_id = str(row.get("execution_id", ""))
        roles: dict[str, Mapping[str, Any]] = {}
        for raw_item in _sequence(row.get("transfer_inputs"), label=f"{execution_id} inputs"):
            item = _mapping(raw_item, label=f"{execution_id} input")
            role = str(item.get("role", ""))
            if role in roles:
                raise ScaffoldContractError(f"{execution_id} duplicates transfer role {role}.")
            roles[role] = item
        result[execution_id] = roles
    if len(result) != context.expected_cell_count or set(result) != set(context.jobs):
        raise ScaffoldContractError("Transfer rows do not cover the inert jobs exactly.")
    return result


def _verified_transfer_binding(
    item: Mapping[str, Any], *, context: FinalizationContext, label: str
) -> dict[str, Any]:
    binding = _plain_binding(item)
    _bound_path(binding=binding, repo_root=context.repo_root, label=label)
    return binding


def _resolve_resumes_and_sources(
    *,
    inputs: Mapping[str, Any],
    context: FinalizationContext,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Mapping[str, Any]]],
]:
    declared = _mapping(
        inputs.get("predecessor_bindings"), label="predecessor bindings"
    )
    if set(declared) != set(context.expected_pending_ids):
        raise ScaffoldContractError("Exactly the nine external predecessor bindings are required.")
    predecessor_bindings: dict[str, Mapping[str, Any]] = {}
    resumes: dict[str, Mapping[str, Any]] = {}
    for execution_id in sorted(context.expected_pending_ids):
        file_binding = _mapping(declared[execution_id], label=f"{execution_id} binding file")
        requirement = context.requirements[execution_id]
        if file_binding.get("path") != requirement.get("binding_path"):
            raise ScaffoldContractError(f"{execution_id} external binding path drifted.")
        _binding_path, binding = _load_bound_controlled_json(
            binding=file_binding,
            repo_root=context.repo_root,
            label=f"{execution_id} predecessor binding",
        )
        validated = validate_predecessor_binding(
            binding=binding,
            requirement=requirement,
            repo_root=context.repo_root,
            rehash_resume=True,
        )
        predecessor_bindings[execution_id] = {
            "file_binding": _plain_binding(file_binding),
            "payload": validated,
        }
        resumes[execution_id] = _mapping(
            validated.get("resume_input"), label=f"{execution_id} resume"
        )
    for execution_id, job in context.jobs.items():
        if execution_id not in resumes:
            resumes[execution_id] = _mapping(
                job.get("resume_input"), label=f"{execution_id} inherited resume"
            )

    roles_by_execution = _transfer_roles(context)
    sources: dict[str, dict[str, Mapping[str, Any]]] = {}
    for execution_id, job in context.jobs.items():
        roles = roles_by_execution[execution_id]
        row_sources: dict[str, Mapping[str, Any]] = {}
        for role in ("source_archive", "source_manifest", "source_delta_receipt"):
            item = _mapping(roles.get(role), label=f"{execution_id} {role}")
            row_sources[role] = _verified_transfer_binding(
                item, context=context, label=f"{execution_id} {role}"
            )
            if role != "source_archive":
                _load_bound_controlled_json(
                    binding=row_sources[role],
                    repo_root=context.repo_root,
                    label=f"{execution_id} {role}",
                )
        expected_sources = _mapping(job.get("effective_sources"), label="job effective sources")
        if any(_plain_binding(expected_sources[role]) != row_sources[role] for role in row_sources):
            raise ScaffoldContractError(f"{execution_id} source transfer relation drifted.")
        resume = resumes[execution_id]
        validate_resume_input_contents(
            resume=resume,
            repo_root=context.repo_root,
            expected_route_contract_sha256=str(
                _mapping(job.get("source_protocol"), label="source protocol").get(
                    "route_contract_sha256", ""
                )
            ),
            expected_depth=SOURCE_HORIZON,
        )
        sources[execution_id] = row_sources
    if len(resumes) != context.expected_cell_count:
        raise ScaffoldContractError("Authenticated resume coverage is incomplete.")
    return predecessor_bindings, resumes, sources


def _final_job(
    *,
    execution_id: str,
    inert_job: Mapping[str, Any],
    inert_job_binding: Mapping[str, Any],
    resume: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
    resource: Mapping[str, Any],
    scaffold_manifest_binding: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(canonical_json_bytes(inert_job))
    result.pop("sha256", None)
    result.update(
        {
            "schema": FINAL_JOB_SCHEMA,
            "package_id": FINAL_PACKAGE_ID,
            "campaign_id": FINAL_CAMPAIGN_ID,
            "status": "authorized_ordinary_held_not_submitted",
            "resume_input": json.loads(canonical_json_bytes(resume)),
            "resume_origin": (
                "external_controlled_cycle_authenticated"
                if predecessor is not None
                else "sealed_v2_inherited_read_only"
            ),
            "predecessor_binding_sha256": (
                predecessor["payload"]["sha256"] if predecessor is not None else None
            ),
            "resume_source": (
                {
                    "predecessor_binding_sha256": predecessor["payload"]["sha256"],
                    "retrieval_completion_receipt": predecessor["payload"][
                        "retrieval_completion_receipt"
                    ],
                    "attempt_archive": predecessor["payload"]["attempt_archive"],
                }
                if predecessor is not None
                else result.get("resume_source")
            ),
            "resources": {
                **resource["approved_envelope"],
                "source": (
                    "authenticated_r50_history_plus_conservative_"
                    "r70_headroom_v1"
                ),
                "r70_demonstration_status": "worst_bucket_pilot_pending",
                "broad_release_authorized": False,
                "resource_bucket": _resource_bucket(inert_job),
                "resource_observation_sha256": resource["receipt"]["sha256"],
            },
            "inert_v2_binding": {
                "package_id": PACKAGE_ID,
                "scaffold_manifest": dict(scaffold_manifest_binding),
                "job": dict(inert_job_binding),
                "job_canonical_sha256": inert_job["sha256"],
            },
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_ready": True,
            "submitted": False,
        }
    )
    if result.get("scientific_settings_sha256") != canonical_sha256(
        result.get("scientific_settings")
    ):
        raise ScaffoldContractError(f"{execution_id} scientific settings drifted in v3.")
    return digested(result)


def _build_jobs_and_control_plane(
    *,
    context: FinalizationContext,
    predecessor_bindings: Mapping[str, Mapping[str, Any]],
    resumes: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
    runtime: Mapping[str, Any],
    image: Mapping[str, Any],
    resources: Mapping[str, Any],
    output_relative_root: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    final_jobs: dict[str, dict[str, Any]] = {}
    job_bindings: dict[str, dict[str, Any]] = {}
    executions: list[dict[str, Any]] = []
    ordered_ids = [str(row["execution_id"]) for row in context.transfer["rows"]]
    for execution_id in ordered_ids:
        inert_job = context.jobs[execution_id]
        bucket = _resource_bucket(inert_job)
        final = _final_job(
            execution_id=execution_id,
            inert_job=inert_job,
            inert_job_binding=context.job_bindings[execution_id],
            resume=resumes[execution_id],
            predecessor=predecessor_bindings.get(execution_id),
            resource=resources["buckets"][bucket],
            scaffold_manifest_binding=context.scaffold_manifest_binding,
        )
        relative = f"{output_relative_root}/{FINAL_JOBS_DIR}/{execution_id}.json"
        binding = _payload_binding(
            relative_path=relative,
            payload=_json_payload(final),
            canonical_digest=final["sha256"],
        )
        final_jobs[execution_id] = final
        job_bindings[execution_id] = binding
        executions.append(
            {
                "execution_id": execution_id,
                "job": binding,
                "scientific_settings_sha256": final["scientific_settings_sha256"],
                "resume_archive": _plain_binding(
                    _mapping(resumes[execution_id].get("archive"), label="resume archive")
                ),
                "sources": sources[execution_id],
                "resource_bucket": bucket,
                "resource_observation": resources["buckets"][bucket]["binding"],
            }
        )
    control_plane = digested(
        {
            "schema": FINAL_CONTROL_PLANE_SCHEMA,
            "inert_package_id": PACKAGE_ID,
            "final_package_id": FINAL_PACKAGE_ID,
            "campaign_id": FINAL_CAMPAIGN_ID,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "inert_scaffold_manifest": dict(context.scaffold_manifest_binding),
            "inert_inventory_sha256": context.inert_inventory_sha256,
            "predecessor_authentication_dependencies": list(
                context.scaffold_manifest.get(
                    "external_control_dependencies", []
                )
            ),
            "runtime_bundle": runtime["bundle"],
            "runtime_bundle_manifest": runtime["manifest"],
            "row_bootstrap": runtime["bootstrap"],
            "image": image["image"],
            "image_verification": image["verification"],
            "resource_evidence": resources["binding"],
            "resource_release_gate": {
                "held_submission": "passed",
                "r70_worst_bucket_pilot": resources[
                    "r70_worst_bucket_pilot"
                ],
                "broad_release_authorized": False,
            },
            "lifecycle": {
                "mode": LIFECYCLE_MODE,
                "initial_state": "all_rows_held",
                "automatic_release": False,
                "release_scope": "exact_cluster_proc_only",
            },
            "execution_count": len(executions),
            "executions": executions,
            "status": "authorization_intent_complete",
        }
    )
    return final_jobs, job_bindings, control_plane


def _validate_authorizations(
    *,
    inputs: Mapping[str, Any],
    context: FinalizationContext,
    final_jobs: Mapping[str, Mapping[str, Any]],
    job_bindings: Mapping[str, Mapping[str, Any]],
    resumes: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Mapping[str, Any]]],
    runtime: Mapping[str, Any],
    image: Mapping[str, Any],
    resources: Mapping[str, Any],
    control_plane: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    declared = _mapping(inputs.get("authorizations"), label="authorizations")
    if set(declared) != set(final_jobs):
        raise ScaffoldContractError("Exactly one authorization per final job is required.")
    lifecycle = {
        "mode": LIFECYCLE_MODE,
        "initial_state": "held",
        "automatic_release": False,
        "release_scope": "exact_cluster_proc_only",
    }
    validated: dict[str, dict[str, Any]] = {}
    for execution_id in final_jobs:
        file_binding = _mapping(declared[execution_id], label=f"{execution_id} authorization binding")
        _path, receipt = _load_bound_controlled_json(
            binding=file_binding,
            repo_root=context.repo_root,
            label=f"{execution_id} authorization",
        )
        bucket = _resource_bucket(final_jobs[execution_id])
        expected = {
            "schema": ACTIVATION_AUTHORIZATION_SCHEMA,
            "status": "passed",
            "authorization_id": f"{FINAL_PACKAGE_ID}::{execution_id}",
            "package_id": FINAL_PACKAGE_ID,
            "inert_package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "job": dict(job_bindings[execution_id]),
            "scientific_settings_sha256": final_jobs[execution_id][
                "scientific_settings_sha256"
            ],
            "resume_archive": _plain_binding(
                _mapping(resumes[execution_id].get("archive"), label="resume archive")
            ),
            "source_archive_sha256": sources[execution_id]["source_archive"]["sha256"],
            "runtime_bundle": runtime["bundle"],
            "runtime_bundle_manifest": runtime["manifest"],
            "row_bootstrap": runtime["bootstrap"],
            "image": image["image"],
            "image_verification": image["verification"],
            "resource_evidence": resources["binding"],
            "resource_observation": resources["buckets"][bucket]["binding"],
            "activation_control_plane_sha256": control_plane["sha256"],
            "lifecycle": lifecycle,
            "execution_authorized": True,
            "submission_authorized": True,
            "release_authorized": False,
            "submission_state": "authorized_not_submitted",
            "remote_stage": False,
            "condor_submit": False,
            "submitted": False,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise ScaffoldContractError(
                    f"{execution_id} authorization relation drifted at {key}."
                )
        if set(receipt) != set(expected) | {"authorized_utc", "sha256"}:
            raise ScaffoldContractError(f"{execution_id} authorization field closure drifted.")
        _utc(receipt.get("authorized_utc"), label=f"{execution_id} authorized_utc")
        validated[execution_id] = {
            "file_binding": _plain_binding(file_binding),
            "payload": receipt,
            "bytes": _json_payload(receipt),
        }
    return validated


def _prepare(
    *,
    context: FinalizationContext,
    activation_inputs_path: Path,
    output_relative_root: str,
    require_authorizations: bool,
) -> dict[str, Any]:
    input_binding, input_bytes, inputs = _load_activation_inputs(
        activation_inputs_path,
        context=context,
        require_authorizations=require_authorizations,
    )
    predecessor_bindings, resumes, sources = _resolve_resumes_and_sources(
        inputs=inputs, context=context
    )
    runtime = _validate_runtime(inputs=inputs, context=context)
    image = _validate_image(inputs=inputs, context=context)
    resources = _validate_resources(inputs=inputs, context=context)
    final_jobs, job_bindings, control_plane = _build_jobs_and_control_plane(
        context=context,
        predecessor_bindings=predecessor_bindings,
        resumes=resumes,
        sources=sources,
        runtime=runtime,
        image=image,
        resources=resources,
        output_relative_root=output_relative_root,
    )
    authorizations = (
        _validate_authorizations(
            inputs=inputs,
            context=context,
            final_jobs=final_jobs,
            job_bindings=job_bindings,
            resumes=resumes,
            sources=sources,
            runtime=runtime,
            image=image,
            resources=resources,
            control_plane=control_plane,
        )
        if require_authorizations
        else {}
    )
    return {
        "inputs": inputs,
        "input_binding": input_binding,
        "input_bytes": input_bytes,
        "predecessor_bindings": predecessor_bindings,
        "resumes": resumes,
        "sources": sources,
        "runtime": runtime,
        "image": image,
        "resources": resources,
        "final_jobs": final_jobs,
        "job_bindings": job_bindings,
        "control_plane": control_plane,
        "authorizations": authorizations,
    }


def render_submit_descriptor(
    *, queue_path: str, batch_name: str, runtime_root: str
) -> str:
    """Render the only allowed row-sharded ordinary-held transfer shape."""

    safe_relative_path(queue_path, label="activation queue path")
    safe_relative_path(runtime_root, label="runtime output root")
    if not batch_name or any(character in batch_name for character in '"\n\r'):
        raise ScaffoldContractError("Batch name is unsafe.")
    descriptor = f"""# Paper-I stationary-core RA r50->r70 v3 ordinary held activation.
universe = vanilla
executable = /bin/bash
transfer_executable = False

arguments = $(row_bootstrap) --runtime-bundle $(runtime_bundle) --job $(job_path) --authorization $(authorization_path) --source-archive $(source_archive) --source-manifest $(source_manifest) --source-delta $(source_delta) --resume-archive $(resume_archive) --image $(image_path) --output transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = $(row_bootstrap),$(runtime_bundle),$(job_path),$(authorization_path),$(source_archive),$(source_manifest),$(source_delta),$(resume_archive),$(image_path)
transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz
transfer_output_remaps = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz={runtime_root}/fetched/$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"

request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
requirements = TARGET.HasSIF

notification = Never
getenv = False
stream_output = False
stream_error = False
+JobBatchName = "{batch_name}"
+HolsteinLifecycleMode = "{LIFECYCLE_MODE}"
hold = True
periodic_release = False
leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)

log = {runtime_root}/logs/$(Cluster).$(Process)__$(execution_id).log
output = {runtime_root}/logs/$(Cluster).$(Process)__$(execution_id).out
error = {runtime_root}/logs/$(Cluster).$(Process)__$(execution_id).err

queue {','.join(QUEUE_VARIABLES)} from {queue_path}
"""
    assert_row_sharded_descriptor(descriptor)
    return descriptor


def assert_row_sharded_descriptor(descriptor: str) -> None:
    transfer_lines = [
        line.strip()
        for line in descriptor.splitlines()
        if line.strip().startswith("transfer_input_files")
    ]
    if len(transfer_lines) != 1:
        raise ScaffoldContractError("Descriptor must have one transfer list.")
    expected = {
        "$(row_bootstrap)",
        "$(runtime_bundle)",
        "$(job_path)",
        "$(authorization_path)",
        "$(source_archive)",
        "$(source_manifest)",
        "$(source_delta)",
        "$(resume_archive)",
        "$(image_path)",
    }
    observed = {
        item.strip() for item in transfer_lines[0].split("=", 1)[1].split(",")
    }
    lowered = descriptor.lower()
    if observed != expected:
        raise ScaffoldContractError("Descriptor transfer list is not the exact row shard.")
    if (
        "hold = true" not in lowered
        or "periodic_release = false" not in lowered
        or "max_materialize" in lowered
        or "max_idle" in lowered
        or "resume_inputs/" in transfer_lines[0].lower()
        or PACKAGE_RELATIVE_ROOT.lower() + "," in transfer_lines[0].lower()
    ):
        raise ScaffoldContractError("Descriptor lifecycle/row-sharding contract drifted.")


def _queue_value(value: Any, *, label: str) -> str:
    result = str(value)
    if not result or any(character in result for character in "\t\n\r\x00"):
        raise ScaffoldContractError(f"Unsafe queue value: {label}")
    return result


def _output_files(
    *,
    prepared: Mapping[str, Any],
    context: FinalizationContext,
    output_relative_root: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    files: dict[str, bytes] = {}
    final_jobs = prepared["final_jobs"]
    authorizations = prepared["authorizations"]
    for execution_id, job in final_jobs.items():
        files[f"{FINAL_JOBS_DIR}/{execution_id}.json"] = _json_payload(job)
        files[f"{FINAL_AUTHORIZATIONS_DIR}/{execution_id}.json"] = authorizations[
            execution_id
        ]["bytes"]
    for execution_id, row in prepared["predecessor_bindings"].items():
        files[f"{FINAL_PREDECESSOR_BINDINGS_DIR}/{execution_id}.json"] = _json_payload(
            row["payload"]
        )
    files[ACTIVATION_INPUT_COPY_NAME] = prepared["input_bytes"]
    files[ACTIVATION_CONTROL_PLANE_NAME] = _json_payload(prepared["control_plane"])

    transfer_rows = context.transfer["rows"]
    queue_lines: list[str] = []
    for row in transfer_rows:
        execution_id = str(row["execution_id"])
        job = final_jobs[execution_id]
        source = prepared["sources"][execution_id]
        resume = _mapping(
            prepared["resumes"][execution_id].get("archive"), label="resume archive"
        )
        values = (
            execution_id,
            f"{output_relative_root}/{FINAL_JOBS_DIR}/{execution_id}.json",
            f"{output_relative_root}/{FINAL_AUTHORIZATIONS_DIR}/{execution_id}.json",
            prepared["runtime"]["bootstrap"]["path"],
            prepared["runtime"]["bundle"]["path"],
            source["source_archive"]["path"],
            source["source_manifest"]["path"],
            source["source_delta_receipt"]["path"],
            resume["path"],
            prepared["image"]["image"]["path"],
            job["resources"]["request_cpus"],
            job["resources"]["request_memory_mb"],
            job["resources"]["request_disk_mb"],
            job["resources"]["max_runtime_seconds"],
        )
        queue_lines.append(
            "\t".join(
                _queue_value(value, label=f"{execution_id} queue field") for value in values
            )
        )
    files[ACTIVATION_QUEUE_NAME] = ("\n".join(queue_lines) + "\n").encode("utf-8")
    descriptor = render_submit_descriptor(
        queue_path=f"{output_relative_root}/{ACTIVATION_QUEUE_NAME}",
        batch_name=FINAL_BATCH_NAME,
        runtime_root=f"{output_relative_root}_runtime",
    )
    files[SUBMIT_NAME] = descriptor.encode("utf-8")

    generated: dict[str, Any] = {}
    for relative, payload in files.items():
        canonical = None
        if relative.endswith(".json"):
            parsed = json.loads(payload)
            canonical = parsed.get("sha256")
        generated[relative] = _payload_binding(
            relative_path=f"{output_relative_root}/{relative}",
            payload=payload,
            canonical_digest=canonical,
        )
    manifest = digested(
        {
            "schema": FINAL_MANIFEST_SCHEMA,
            "package_id": FINAL_PACKAGE_ID,
            "campaign_id": FINAL_CAMPAIGN_ID,
            "inert_package_id": PACKAGE_ID,
            "run_class": "paper_facing",
            "execution_target": "chtc",
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "inert_scaffold_manifest": dict(context.scaffold_manifest_binding),
            "inert_inventory_sha256": context.inert_inventory_sha256,
            "predecessor_authentication_dependencies": list(
                context.scaffold_manifest.get(
                    "external_control_dependencies", []
                )
            ),
            "activation_inputs_source": prepared["input_binding"],
            "activation_control_plane_sha256": prepared["control_plane"]["sha256"],
            "execution_count": len(final_jobs),
            "new_predecessor_binding_count": len(prepared["predecessor_bindings"]),
            "runtime_bundle": prepared["runtime"]["bundle"],
            "runtime_bundle_manifest": prepared["runtime"]["manifest"],
            "image": prepared["image"]["image"],
            "image_verification": prepared["image"]["verification"],
            "resource_evidence": prepared["resources"]["binding"],
            "resource_release_gate": {
                "held_submission": "passed",
                "r70_worst_bucket_pilot": prepared["resources"][
                    "r70_worst_bucket_pilot"
                ],
                "broad_release_authorized": False,
            },
            "jobs": [generated[f"{FINAL_JOBS_DIR}/{execution_id}.json"] for execution_id in final_jobs],
            "authorizations": [
                generated[f"{FINAL_AUTHORIZATIONS_DIR}/{execution_id}.json"]
                for execution_id in final_jobs
            ],
            "predecessor_bindings": [
                generated[f"{FINAL_PREDECESSOR_BINDINGS_DIR}/{execution_id}.json"]
                for execution_id in sorted(prepared["predecessor_bindings"])
            ],
            "generated": generated,
            "transfer_shape": "one_exact_resume_archive_per_row",
            "aggregate_resume_directory_transferred": False,
            "operational_mode": LIFECYCLE_MODE,
            "initial_state": "all_rows_held",
            "automatic_release": False,
            "release_scope": "exact_cluster_proc_only",
            "execution_authorized": True,
            "submission_authorized": True,
            "release_authorized": False,
            "submission_state": "authorized_not_submitted",
            "remote_stage": False,
            "condor_submit": False,
            "submitted": False,
            "paper_evidence_adopted": False,
            "status": "passed_atomic_v3_ordinary_held_not_submitted",
        }
    )
    files[ACTIVATION_MANIFEST_NAME] = _json_payload(manifest)
    return files, manifest


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ScaffoldContractError(f"Refusing to overwrite staged path: {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_staged(
    *, staging: Path, files: Mapping[str, bytes], manifest: Mapping[str, Any]
) -> None:
    observed = {
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if observed != set(files):
        raise ScaffoldContractError("Staged v3 file closure drifted.")
    for relative, payload in files.items():
        path = staging / relative
        if path.is_symlink() or path.read_bytes() != payload:
            raise ScaffoldContractError(f"Staged v3 bytes drifted: {relative}")
    verify_self_digest(manifest, label="staged v3 manifest")
    assert_row_sharded_descriptor((staging / SUBMIT_NAME).read_text(encoding="utf-8"))
    if len((staging / ACTIVATION_QUEUE_NAME).read_text().splitlines()) != manifest["execution_count"]:
        raise ScaffoldContractError("Staged v3 queue cardinality drifted.")
    for relative, binding in manifest["generated"].items():
        path = staging / relative
        if sha256_file(path) != binding["sha256"] or path.stat().st_size != binding["size_bytes"]:
            raise ScaffoldContractError(f"Staged generated binding drifted: {relative}")


def _publish_atomic(
    *,
    files: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    context: FinalizationContext,
    output_dir: Path,
) -> None:
    parent = output_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ScaffoldContractError("V3 sibling parent is missing or unsafe.")
    if output_dir.exists() or output_dir.is_symlink():
        raise ScaffoldContractError(f"Refusing to overwrite v3 sibling: {output_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=parent))
    renamed = False
    try:
        for relative, payload in files.items():
            _write_new(staging / safe_relative_path(relative, label="staged output path"), payload)
        _validate_staged(staging=staging, files=files, manifest=manifest)
        if _file_inventory_digest(context.package_dir) != context.inert_inventory_sha256:
            raise ScaffoldContractError("Inert v2 changed while v3 was being finalized.")
        os.rename(staging, output_dir)
        renamed = True
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if not renamed and staging.exists():
            shutil.rmtree(staging)


def prepare_activation_intent(
    *, activation_inputs_path: Path, context: FinalizationContext | None = None
) -> dict[str, Any]:
    """Validate evidence and return deterministic job/control-plane digests."""

    active = context or _production_context()
    prepared = _prepare(
        context=active,
        activation_inputs_path=activation_inputs_path,
        output_relative_root=FINAL_PACKAGE_RELATIVE_ROOT,
        require_authorizations=False,
    )
    return {
        "status": "passed_authorization_intent_not_materialized",
        "inert_package_id": PACKAGE_ID,
        "final_package_id": FINAL_PACKAGE_ID,
        "activation_control_plane_sha256": prepared["control_plane"]["sha256"],
        "required_authorization_schema": ACTIVATION_AUTHORIZATION_SCHEMA,
        "execution_count": len(prepared["final_jobs"]),
        "executions": [
            {
                "execution_id": execution_id,
                "job": prepared["job_bindings"][execution_id],
                "scientific_settings_sha256": prepared["final_jobs"][execution_id][
                    "scientific_settings_sha256"
                ],
                "resume_archive": _plain_binding(
                    prepared["resumes"][execution_id]["archive"]
                ),
            }
            for execution_id in prepared["final_jobs"]
        ],
        "v3_materialized": False,
        "submitted": False,
    }


def _build_activation_from_context(
    *,
    activation_inputs_path: Path,
    context: FinalizationContext,
    output_dir: Path,
    output_relative_root: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ScaffoldContractError(f"Refusing to overwrite v3 sibling: {output_dir}")
    prepared = _prepare(
        context=context,
        activation_inputs_path=activation_inputs_path,
        output_relative_root=output_relative_root,
        require_authorizations=True,
    )
    files, manifest = _output_files(
        prepared=prepared,
        context=context,
        output_relative_root=output_relative_root,
    )
    _publish_atomic(
        files=files,
        manifest=manifest,
        context=context,
        output_dir=output_dir,
    )
    return {
        "status": "materialized_atomic_v3_ordinary_held_not_submitted",
        "inert_package_id": PACKAGE_ID,
        "package_id": FINAL_PACKAGE_ID,
        "row_count": len(prepared["final_jobs"]),
        "new_predecessor_binding_count": len(prepared["predecessor_bindings"]),
        "activation_manifest_sha256": manifest["sha256"],
        "output": output_relative_root,
        "submit_descriptor": f"{output_relative_root}/{SUBMIT_NAME}",
        "aggregate_resume_directory_transferred": False,
        "initial_state": "all_rows_held",
        "broad_release_authorized": False,
        "next_gate": "separately_authorized_single_pauli_word_v1_nph7_r70_pilot",
        "condor_submit": False,
        "submitted": False,
    }


def build_activation(*, activation_inputs_path: Path) -> dict[str, Any]:
    """Consume canonical v2 plus external evidence and publish canonical v3."""

    context = _production_context()
    output_dir = context.repo_root / FINAL_PACKAGE_RELATIVE_ROOT
    return _build_activation_from_context(
        activation_inputs_path=activation_inputs_path,
        context=context,
        output_dir=output_dir,
        output_relative_root=FINAL_PACKAGE_RELATIVE_ROOT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-inputs", type=Path, required=True)
    parser.add_argument(
        "--prepare-intent",
        action="store_true",
        help="Validate all non-authorization evidence and print exact job digests only.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = (
            prepare_activation_intent(
                activation_inputs_path=args.activation_inputs.resolve()
            )
            if args.prepare_intent
            else build_activation(
                activation_inputs_path=args.activation_inputs.resolve()
            )
        )
    except (OSError, ScaffoldContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
