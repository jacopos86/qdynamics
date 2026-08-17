#!/usr/bin/env python3
"""Validate one fetched cluster-9650825 L3 worker attempt archive.

This is a local retrieval boundary.  It reads the immutable v3 package and the
submitted activation authority; it neither selects paper evidence nor changes
the package, activation, scheduler, or fetched archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = (
    SCRIPT_DIR
    / "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc"
)
PACKAGE_ID = (
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc"
)
CAMPAIGN_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v3"
BUNDLE_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v3"
PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_l3_weak_holstein_matched_package_manifest_v1"
)
PACKAGE_MANIFEST_SHA256 = (
    "da24fd9467318dcf5786883104945decee3b57d460f3097cd7995ccc64268edf"
)
JOB_SCHEMA = "paper_i_l3_weak_holstein_matched_job_v1"
ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_l3_weak_holstein_activation_request_v1"
)
ACTIVATION_MANIFEST_SCHEMA = (
    "paper_i_l3_weak_holstein_activation_manifest_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_l3_weak_holstein_execution_authorization_v1"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_worker_receipt_v1"
)
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_execution_manifest_v1"
)
VALIDATION_SCHEMA = (
    "paper_i_l3_weak_holstein_cluster9650825_attempt_validation_v1"
)
EXPECTED_CLUSTER_ID = 9_650_825
EXPECTED_CONTROLLER_ROUNDS = 50
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
# These four values are copied from the immutable local submission receipt for
# cluster 9650825.  They bind retrieval to the exact activation that produced
# the submitted queue rather than to any newly self-digested authority.
ACTIVATION_MANIFEST_SHA256 = (
    "0272a8ab052ffcad660c89efdafee52644968ee5a05ab644b702afaa7182ff09"
)
ACTIVATION_MANIFEST_FILE_SHA256 = (
    "d6780219467b6b5c60cf102306890d178e4ac2e205116cb11f9615c531ed2608"
)
ACTIVATION_REQUEST_SHA256 = (
    "2299276b2fc4295f2bc917d99f88c48de55dbf6ff198757ce44db9cb587504bd"
)
ACTIVATION_REQUEST_FILE_SHA256 = (
    "071acebda4db69d7f4b1ee7aeb1e3f21ff0c9b70e760ad732b4ed0b314a27a06"
)
ARTIFACT_ROLES = (
    "execution_manifest",
    "checkpoint",
    "estimator_ledger",
    "result",
    "summary",
)
JSON_ARTIFACT_ROLES = (
    "result",
    "summary",
    "checkpoint",
    "estimator_ledger",
)


class AttemptValidationError(ValueError):
    """Raised when a fetched attempt fails the local retrieval contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    if value.get("sha256") != observed:
        raise AttemptValidationError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptValidationError(f"{label} must be a JSON object.")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AttemptValidationError(f"{label} must be a list.")
    return value


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AttemptValidationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptValidationError(f"{label} is unreadable JSON.") from exc
    return dict(_mapping(value, label=label))


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AttemptValidationError(f"{label} must be a relative path.")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise AttemptValidationError(f"{label} is unsafe.")
    parts = value.split("/")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in parts) or path.is_absolute():
        raise AttemptValidationError(f"{label} is unsafe.")
    return path


def _load_bound_json(
    *,
    root: Path,
    raw_binding: Any,
    expected_path: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = dict(_mapping(raw_binding, label=f"{label} binding"))
    if (
        set(binding) != {"path", "sha256", "size_bytes", "canonical_sha256"}
        or binding.get("path") != expected_path
    ):
        raise AttemptValidationError(f"{label} binding shape/path drifted.")
    relative = _safe_relative_path(binding["path"], label=f"{label} path")
    target = root.joinpath(*relative.parts)
    payload = _load_json_file(target, label=label)
    canonical_digest = verify_self_digest(payload, label=label)
    if (
        target.stat().st_size != binding.get("size_bytes")
        or sha256_file(target) != binding.get("sha256")
        or canonical_digest != binding.get("canonical_sha256")
    ):
        raise AttemptValidationError(f"{label} byte/canonical binding drifted.")
    return payload, binding


def _safe_member_name(name: str) -> str | None:
    if name in {".", "./"}:
        return None
    if (
        not name
        or name.startswith("/")
        or name.startswith("../")
        or "//" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise AttemptValidationError(f"Unsafe attempt member: {name!r}.")
    trimmed = name[2:] if name.startswith("./") else name
    components = trimmed.split("/")
    path = PurePosixPath(trimmed)
    if (
        not path.parts
        or any(part in {"", ".", ".."} for part in components)
        or path.is_absolute()
    ):
        raise AttemptValidationError(f"Unsafe attempt member: {name!r}.")
    return path.as_posix()


def _member_bytes(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise AttemptValidationError(f"{label} has no bytes.")
    return source.read()


def _json_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            _member_bytes(archive, member, label=label).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptValidationError(f"{label} is unreadable JSON.") from exc
    return dict(_mapping(value, label=label))


def _member_sha256_and_size(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    label: str,
) -> tuple[str, int]:
    source = archive.extractfile(member)
    if source is None:
        raise AttemptValidationError(f"{label} has no bytes.")
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _queue_execution_ids() -> tuple[str, ...]:
    queue_path = PACKAGE_DIR / "queue.tsv"
    if not queue_path.is_file() or queue_path.is_symlink():
        raise AttemptValidationError("The sealed package queue is unavailable.")
    rows = queue_path.read_text(encoding="utf-8").splitlines()
    execution_ids = tuple(row.split("\t", 1)[0] for row in rows if row)
    if len(execution_ids) != 6 or len(set(execution_ids)) != 6:
        raise AttemptValidationError("The sealed six-row queue drifted.")
    return execution_ids


def _load_package_and_job(
    execution_id: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    manifest = _load_json_file(manifest_path, label="sealed package manifest")
    verify_self_digest(manifest, label="sealed package manifest")
    execution_ids = _queue_execution_ids()
    if execution_id not in execution_ids:
        raise AttemptValidationError(
            "Expected execution ID is outside the sealed six-row queue."
        )
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_matched_six_cell"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != list(execution_ids)
    ):
        raise AttemptValidationError("Sealed package identity drifted.")
    bindings = [
        row
        for row in _list(manifest.get("jobs"), label="package job bindings")
        if isinstance(row, Mapping) and row.get("execution_id") == execution_id
    ]
    if len(bindings) != 1:
        raise AttemptValidationError("Expected job binding is not unique.")
    binding = bindings[0]
    expected_job_path = f"jobs/{execution_id}.json"
    if binding.get("path") != expected_job_path:
        raise AttemptValidationError("Expected job path drifted.")
    job_path = PACKAGE_DIR / expected_job_path
    job = _load_json_file(job_path, label="sealed job")
    canonical_digest = verify_self_digest(job, label="sealed job")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") != execution_id
        or job.get("target_horizon") != EXPECTED_CONTROLLER_ROUNDS
        or binding.get("canonical_sha256") != canonical_digest
        or binding.get("sha256") != sha256_file(job_path)
        or binding.get("size_bytes") != job_path.stat().st_size
    ):
        raise AttemptValidationError("Sealed job binding drifted.")
    return manifest, job, execution_ids.index(execution_id)


def _validate_activation_authority(
    *,
    activation_manifest_path: Path,
    authorization_path: Path,
    execution_ids: tuple[str, ...],
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation_manifest = _load_json_file(
        activation_manifest_path, label="activation manifest"
    )
    activation_manifest_digest = verify_self_digest(
        activation_manifest, label="activation manifest"
    )
    if (
        activation_manifest_path.name != "activation_manifest.json"
        or activation_manifest_digest != ACTIVATION_MANIFEST_SHA256
        or sha256_file(activation_manifest_path)
        != ACTIVATION_MANIFEST_FILE_SHA256
    ):
        raise AttemptValidationError(
            "Activation manifest is not the exact submitted authority."
        )
    activation_root = activation_manifest_path.parent
    if not activation_root.is_dir() or activation_root.is_symlink():
        raise AttemptValidationError("Activation root is missing or unsafe.")
    request, request_binding = _load_bound_json(
        root=activation_root,
        raw_binding=activation_manifest.get("activation_request"),
        expected_path="activation_request.json",
        label="activation request",
    )
    image_probe, probe_binding = _load_bound_json(
        root=activation_root,
        raw_binding=activation_manifest.get("image_runtime_probe"),
        expected_path="image_runtime_probe.json",
        label="image runtime probe",
    )
    if (
        request_binding.get("canonical_sha256") != ACTIVATION_REQUEST_SHA256
        or request_binding.get("sha256") != ACTIVATION_REQUEST_FILE_SHA256
    ):
        raise AttemptValidationError(
            "Activation request is not the exact submitted request."
        )

    raw_authorizations = _list(
        activation_manifest.get("authorizations"),
        label="activation authorizations",
    )
    authorization_bindings: dict[str, dict[str, Any]] = {}
    for raw in raw_authorizations:
        row = dict(_mapping(raw, label="activation authorization binding"))
        execution_id = str(row.get("execution_id", ""))
        expected_path = f"authorizations/{execution_id}.json"
        if (
            set(row)
            != {
                "execution_id",
                "path",
                "sha256",
                "size_bytes",
                "canonical_sha256",
            }
            or execution_id in authorization_bindings
            or row.get("path") != expected_path
            or execution_id not in execution_ids
        ):
            raise AttemptValidationError(
                "Activation authorization closure drifted."
            )
        _safe_relative_path(row["path"], label="authorization path")
        authorization_bindings[execution_id] = row
    if (
        tuple(authorization_bindings) != execution_ids
        or len(authorization_bindings) != 6
    ):
        raise AttemptValidationError("Activation authorization closure drifted.")

    execution_id = str(job.get("execution_id"))
    authorization_binding = authorization_bindings[execution_id]
    expected_authorization_path = activation_root.joinpath(
        *_safe_relative_path(
            authorization_binding["path"], label="authorization path"
        ).parts
    )
    if authorization_path.absolute() != expected_authorization_path.absolute():
        raise AttemptValidationError(
            "Authorization path is outside its activation-manifest binding."
        )
    authorization, observed_authorization_binding = _load_bound_json(
        root=activation_root,
        raw_binding={
            key: value
            for key, value in authorization_binding.items()
            if key != "execution_id"
        },
        expected_path=str(authorization_binding["path"]),
        label="execution authorization",
    )
    if observed_authorization_binding != {
        key: value
        for key, value in authorization_binding.items()
        if key != "execution_id"
    }:
        raise AttemptValidationError("Execution authorization binding drifted.")

    source_archive = _mapping(
        manifest.get("source_archive"), label="package source archive"
    )
    probe = image_probe.get("pinned_image_runtime_probe")
    backend_probe = probe.get("probe") if isinstance(probe, Mapping) else None
    if (
        activation_manifest.get("schema") != ACTIVATION_MANIFEST_SCHEMA
        or activation_manifest.get("status")
        != "passed_activation_prepared_no_submission"
        or activation_manifest.get("package_id") != PACKAGE_ID
        or activation_manifest.get("campaign_id") != CAMPAIGN_ID
        or activation_manifest.get("bundle_id") != BUNDLE_ID
        or activation_manifest.get("package_manifest_sha256")
        != manifest.get("sha256")
        or activation_manifest.get("activation_request") != request_binding
        or activation_manifest.get("image_runtime_probe") != probe_binding
        or activation_manifest.get("pinned_image_path") != REMOTE_IMAGE_PATH
        or activation_manifest.get("pinned_image_sha256")
        != REMOTE_IMAGE_SHA256
        or activation_manifest.get("authorization_count") != 6
        or activation_manifest.get("launch_ready") is not True
        or activation_manifest.get("execution_authorized") is not True
        or activation_manifest.get("submission_authorized") is not True
        or activation_manifest.get("paper_evidence_adoption_authorized")
        is not False
        or activation_manifest.get("submitted") is not False
        or activation_manifest.get("remote_stage") is not False
        or activation_manifest.get("condor_submit") is not False
        or request.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or request.get("package_id") != PACKAGE_ID
        or request.get("campaign_id") != CAMPAIGN_ID
        or request.get("bundle_id") != BUNDLE_ID
        or request.get("package_manifest_sha256") != manifest.get("sha256")
        or request.get("requested_execution_ids") != list(execution_ids)
        or request.get("scope")
        != "prepare_matched_six_cell_chtc_execution_and_submission_v1"
        or request.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or request.get("explicit_user_authority_recorded") is not True
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("paper_evidence_adoption_authorized") is not False
        or request.get("submitted") is not False
        or image_probe.get("status") != "passed_inert_package"
        or image_probe.get("package_manifest_sha256") != manifest.get("sha256")
        or image_probe.get("launch_ready") is not True
        or image_probe.get("execution_authorized") is not False
        or image_probe.get("submission_authorized") is not False
        or not isinstance(probe, Mapping)
        or probe.get("status") != "passed"
        or probe.get("image_sha256") != REMOTE_IMAGE_SHA256
        or not isinstance(backend_probe, Mapping)
        or backend_probe.get("resolved_backend_name") != "FakeMarrakesh"
        or backend_probe.get("backend_resolution_kind") != "fake_exact"
    ):
        raise AttemptValidationError("Activation authority binding drifted.")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("bundle_id") != BUNDLE_ID
        or authorization.get("execution_id") != job.get("execution_id")
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != manifest.get("sha256")
        or authorization.get("protocol_sha256")
        != job.get("protocol_sha256")
        or authorization.get("source_archive_sha256")
        != source_archive.get("sha256")
        or authorization.get("activation_request") != request_binding
        or authorization.get("image_runtime_probe") != probe_binding
        or authorization.get("pinned_image_path") != REMOTE_IMAGE_PATH
        or authorization.get("pinned_image_sha256") != REMOTE_IMAGE_SHA256
        or authorization.get("scope") != "single_cell_chtc_execution_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("submitted") is not False
    ):
        raise AttemptValidationError("Execution authorization binding drifted.")
    return activation_manifest, authorization
def _expected_artifact_paths(job: Mapping[str, Any]) -> dict[str, str]:
    raw = _mapping(
        job.get("expected_run_artifacts"), label="expected run artifacts"
    )
    if set(raw) != set(ARTIFACT_ROLES):
        raise AttemptValidationError("Expected artifact role closure drifted.")
    execution_id = str(job["execution_id"])
    suffixes = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    result: dict[str, str] = {}
    for role, suffix in suffixes.items():
        row = _mapping(raw.get(role), label=f"expected {role}")
        expected = f"runs/{execution_id}/{suffix}"
        if (
            row.get("path") != expected
            or row.get("required") is not True
            or row.get("direct_file_required") is not True
            or row.get("reference_receipt_required") is not False
            or row.get("fulfillment_kind") != "direct_execution_v1"
        ):
            raise AttemptValidationError(
                f"Expected artifact contract drifted for {role}."
            )
        result[role] = expected
    return result


def validate_attempt(
    *,
    archive_path: Path,
    expected_execution_id: str,
    activation_manifest_path: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    """Validate one exact fetched attempt without extracting or adopting it.

    The cluster/proc assertion is scheduler-side provenance: v3 records it in
    the transferred archive basename and queue position, but does not attest
    those scheduler IDs inside the worker receipt or execution manifest.
    """

    if not archive_path.is_file() or archive_path.is_symlink():
        raise AttemptValidationError("Fetched attempt archive is unavailable/unsafe.")
    manifest, job, expected_proc_id = _load_package_and_job(
        expected_execution_id
    )
    expected_name = (
        rf"{re.escape(expected_execution_id)}__"
        rf"(?P<cluster>[0-9]+)__(?P<proc>[0-9]+)\.tar\.gz"
    )
    match = re.fullmatch(expected_name, archive_path.name)
    if (
        match is None
        or int(match.group("cluster")) != EXPECTED_CLUSTER_ID
        or int(match.group("proc")) != expected_proc_id
    ):
        raise AttemptValidationError(
            "Attempt filename is not the exact cluster/proc/execution mapping."
        )
    execution_ids = _queue_execution_ids()
    activation_manifest, authorization = _validate_activation_authority(
        activation_manifest_path=activation_manifest_path,
        authorization_path=authorization_path,
        execution_ids=execution_ids,
        manifest=manifest,
        job=job,
    )
    expected_paths = _expected_artifact_paths(job)
    expected_files = {
        "worker_exit_status.txt",
        "worker_receipt.json",
        *expected_paths.values(),
    }
    allowed_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }

    try:
        archive = tarfile.open(archive_path, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise AttemptValidationError("Fetched attempt is not a readable tar.gz.") from exc
    with archive:
        files: dict[str, tarfile.TarInfo] = {}
        directories: set[str] = set()
        root_seen = False
        for member in archive:
            name = _safe_member_name(member.name)
            if member.issym() or member.islnk():
                raise AttemptValidationError(
                    f"Attempt links are forbidden: {member.name!r}."
                )
            if name is None:
                if not member.isdir() or root_seen:
                    raise AttemptValidationError("Attempt root member drifted.")
                root_seen = True
                continue
            if member.isdir():
                if name in directories or name not in allowed_directories:
                    raise AttemptValidationError(
                        f"Unsafe/duplicate attempt directory: {name}."
                    )
                directories.add(name)
                continue
            if not member.isfile() or name in files:
                raise AttemptValidationError(
                    f"Unsafe/duplicate attempt member: {name}."
                )
            files[name] = member
        if set(files) != expected_files:
            raise AttemptValidationError(
                "Attempt regular-file closure drifted: "
                f"missing={sorted(expected_files - set(files))}, "
                f"extra={sorted(set(files) - expected_files)}."
            )

        status_bytes = _member_bytes(
            archive,
            files["worker_exit_status.txt"],
            label="worker exit status",
        )
        try:
            worker_exit_status = int(status_bytes.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise AttemptValidationError("Worker exit status is invalid.") from exc
        if worker_exit_status != 0:
            raise AttemptValidationError(
                f"Worker exit status is nonzero: {worker_exit_status}."
            )

        receipt = _json_member(
            archive,
            files["worker_receipt.json"],
            label="worker receipt",
        )
        verify_self_digest(receipt, label="worker receipt")
        if (
            receipt.get("schema") != WORKER_RECEIPT_SCHEMA
            or receipt.get("status") != "passed"
            or receipt.get("package_id") != PACKAGE_ID
            or receipt.get("campaign_id") != CAMPAIGN_ID
            or receipt.get("execution_id") != expected_execution_id
            or receipt.get("job_spec_sha256") != job.get("sha256")
            or receipt.get("authorization_sha256")
            != authorization.get("sha256")
            or receipt.get("controller_rounds_completed")
            != EXPECTED_CONTROLLER_ROUNDS
            or receipt.get("fresh_start") is not True
        ):
            raise AttemptValidationError("Worker receipt binding drifted.")

        raw_artifacts = _list(
            receipt.get("artifacts"), label="worker receipt artifacts"
        )
        artifact_bindings: dict[str, Mapping[str, Any]] = {}
        for raw in raw_artifacts:
            row = _mapping(raw, label="worker artifact binding")
            path = str(row.get("path", ""))
            if (
                set(row) != {"path", "sha256", "size_bytes"}
                or path in artifact_bindings
            ):
                raise AttemptValidationError("Worker artifact binding drifted.")
            artifact_bindings[path] = row
        if set(artifact_bindings) != set(expected_paths.values()):
            raise AttemptValidationError("Worker artifact receipt closure drifted.")

        actual_bindings: dict[str, dict[str, Any]] = {}
        for role, path in expected_paths.items():
            digest, size = _member_sha256_and_size(
                archive, files[path], label=f"{role} artifact"
            )
            binding = artifact_bindings[path]
            if (
                binding.get("sha256") != digest
                or binding.get("size_bytes") != size
            ):
                raise AttemptValidationError(
                    f"Artifact hash/size binding drifted for {role}."
                )
            actual_bindings[role] = {
                "path": path,
                "sha256": digest,
                "size_bytes": size,
            }

        execution_manifest = _json_member(
            archive,
            files[expected_paths["execution_manifest"]],
            label="execution manifest",
        )
        verify_self_digest(execution_manifest, label="execution manifest")
        if (
            execution_manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
            or execution_manifest.get("status") != "passed"
            or execution_manifest.get("package_id") != PACKAGE_ID
            or execution_manifest.get("campaign_id") != CAMPAIGN_ID
            or execution_manifest.get("execution_id") != expected_execution_id
            or execution_manifest.get("job_spec_sha256") != job.get("sha256")
            or execution_manifest.get("authorization_sha256")
            != authorization.get("sha256")
            or execution_manifest.get("protocol_sha256")
            != job.get("protocol_sha256")
            or execution_manifest.get("route_contract_sha256")
            != job.get("route_contract_sha256")
            or execution_manifest.get("execution_entrypoint")
            != job.get("execution_entrypoint")
            or execution_manifest.get("target_horizon")
            != EXPECTED_CONTROLLER_ROUNDS
            or execution_manifest.get("controller_rounds_completed")
            != EXPECTED_CONTROLLER_ROUNDS
            or execution_manifest.get("fresh_start") is not True
            or execution_manifest.get("source_checkpoint_consumed") is not False
            or receipt.get("execution_manifest_sha256")
            != execution_manifest.get("sha256")
        ):
            raise AttemptValidationError("Execution manifest binding drifted.")

        output_payloads = _mapping(
            execution_manifest.get("output_payloads"),
            label="execution manifest output payloads",
        )
        payload_roles = set(ARTIFACT_ROLES) - {"execution_manifest"}
        if set(output_payloads) != payload_roles:
            raise AttemptValidationError(
                "Execution manifest payload-role closure drifted."
            )
        for role in payload_roles:
            row = _mapping(
                output_payloads.get(role), label=f"execution payload {role}"
            )
            if dict(row) != actual_bindings[role]:
                raise AttemptValidationError(
                    f"Execution manifest artifact binding drifted for {role}."
                )

        for role in JSON_ARTIFACT_ROLES:
            _json_member(
                archive,
                files[expected_paths[role]],
                label=f"{role} artifact",
            )

    return digested(
        {
            "schema": VALIDATION_SCHEMA,
            "status": "passed_validated_no_adoption",
            "cluster_id": EXPECTED_CLUSTER_ID,
            "proc_id": expected_proc_id,
            "execution_id": expected_execution_id,
            "archive_path": archive_path.resolve().as_posix(),
            "archive_sha256": sha256_file(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "package_manifest_sha256": manifest["sha256"],
            "job_spec_sha256": job["sha256"],
            "activation_manifest_sha256": activation_manifest["sha256"],
            "authorization_sha256": authorization["sha256"],
            "worker_receipt_sha256": receipt["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": EXPECTED_CONTROLLER_ROUNDS,
            "artifact_bindings": [
                actual_bindings[role] for role in ARTIFACT_ROLES
            ],
            "scheduler_identity_provenance": {
                "kind": "archive_basename_and_sealed_queue_position_v1",
                "cluster_proc_attested_inside_archive": False,
                "limitation": (
                    "v3 worker receipt and execution manifest do not carry "
                    "scheduler cluster/proc IDs"
                ),
            },
            "automatic_attempt_selection_performed": False,
            "paper_evidence_adopted": False,
            "external_state_changed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-execution-id", required=True)
    parser.add_argument("--activation-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = validate_attempt(
            archive_path=args.archive,
            expected_execution_id=str(args.expected_execution_id),
            activation_manifest_path=args.activation_manifest,
            authorization_path=args.authorization,
        )
    except (OSError, tarfile.TarError, AttemptValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
