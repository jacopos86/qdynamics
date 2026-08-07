#!/usr/bin/env python3
"""Derive quota-safe v2 subset packages from the two sealed v1 campaigns.

The scientific parent jobs, protocols, runner, and package contract are copied
byte-for-byte.  The source archive changes exactly one member:
``pipelines/static_adapt/current_checkpoint.py``.  The replacement only retires
authenticated predecessor sidecars after a successor checkpoint is durable.

This builder is local-only.  It does not stage, submit, remove, hold, release,
or otherwise contact CHTC.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parents[1]
CHECKPOINT_MEMBER = "pipelines/static_adapt/current_checkpoint.py"
PARENT_CHECKPOINT_SHA256 = (
    "16ffddfdbf20674c50af7b797131efa40478c5281d16f4f034d7db49b8249cb8"
)
REPAIRED_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ROLLING_RECEIPT_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_9395481_9395482_rolling_retrieval_20260730_receipt.json"
)
PRESERVATION_SNAPSHOT_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_completed_v1_preservation_"
    "9395481_9395482_20260730_v1.json"
)
PRESERVATION_SNAPSHOT_SCHEMA = (
    "paper_i_ra_adapt_completed_v1_preservation_snapshot_v1"
)
RETIREMENT_RECEIPT_SCHEMA = (
    "paper_i_checkpoint_retention_parent_cluster_retirement_v1"
)


class RepairPackageError(ValueError):
    """Raised when the v1-to-v2 derivation contract is not exact."""


@dataclass(frozen=True)
class Campaign:
    key: str
    parent_package_dirname: str
    parent_activation_dirname: str
    parent_submission_receipt_name: str
    parent_cluster_id: int
    completed_proc_ids: tuple[int, ...]
    package_dirname: str
    activation_dirname: str
    operational_package_id: str
    activation_id: str
    batch_name: str
    campaign_cell_count: int
    max_materialize: int
    expected_source_archive_sha256: str | None = None

    @property
    def parent_package(self) -> Path:
        return BASE / self.parent_package_dirname

    @property
    def parent_activation(self) -> Path:
        return BASE / self.parent_activation_dirname

    @property
    def parent_submission_receipt(self) -> Path:
        return BASE / self.parent_submission_receipt_name

    @property
    def release_activation_dirname(self) -> str:
        return f"{self.activation_dirname}_release_v1"

    @property
    def release_activation_id(self) -> str:
        return f"{self.activation_id}_release_v1"

    @property
    def ordinary_held_release_activation_dirname(self) -> str:
        return f"{self.activation_dirname}_release_v2"

    @property
    def ordinary_held_release_activation_id(self) -> str:
        return f"{self.activation_id}_release_v2"

    @property
    def ordinary_held_release_batch_name(self) -> str:
        return f"{self.batch_name}-ordinary-held-release-v2"


CAMPAIGNS = (
    Campaign(
        key="factorial",
        parent_package_dirname=(
            "ra_always_factorial48_r50_20260730_v1_chtc"
        ),
        parent_activation_dirname=(
            "ra_always_factorial48_r50_20260730_v1_chtc_activation"
        ),
        parent_submission_receipt_name=(
            "ra_always_factorial48_r50_20260730_v1_chtc_"
            "submission_receipt.json"
        ),
        parent_cluster_id=9395481,
        completed_proc_ids=(0, 1, 3),
        package_dirname=(
            "ra_always_factorial48_r50_20260730_v2_chtc"
        ),
        activation_dirname=(
            "ra_always_factorial48_r50_20260730_v2_chtc_activation"
        ),
        operational_package_id=(
            "paper_i_ra_adapt_always_factorial48_r50_"
            "20260730_v2_chtc"
        ),
        activation_id=(
            "paper_i_ra_adapt_always_factorial48_r50_"
            "20260730_v2_chtc_activation_v1"
        ),
        batch_name=(
            "paper-i-ra-adapt-always-factorial48-r50-"
            "20260730-v2-checkpoint-retention"
        ),
        campaign_cell_count=48,
        max_materialize=1,
    ),
    Campaign(
        key="global_singleton",
        parent_package_dirname=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v1_chtc"
        ),
        parent_activation_dirname=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v1_chtc_activation"
        ),
        parent_submission_receipt_name=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v1_chtc_submission_receipt.json"
        ),
        parent_cluster_id=9395482,
        completed_proc_ids=(0,),
        package_dirname=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v2_chtc"
        ),
        activation_dirname=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v2_chtc_activation"
        ),
        operational_package_id=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v2_chtc"
        ),
        activation_id=(
            "paper_i_ra_adapt_global_singleton_insertion12_"
            "r50_20260730_v2_chtc_activation_v1"
        ),
        batch_name=(
            "paper-i-ra-adapt-global-singleton-insertion12-"
            "r50-20260730-v2-checkpoint-retention"
        ),
        campaign_cell_count=12,
        max_materialize=1,
        expected_source_archive_sha256=(
            "4c79f8de78c1700120f2018b098d361c"
            "37c3b054e261be7214cb5fb74d862dd8"
        ),
    ),
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise RepairPackageError("Self digest input already has sha256.")
    result["sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairPackageError(f"Could not load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RepairPackageError(f"{label} is not a JSON object: {path}")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    observed = payload.get("sha256")
    projection = dict(payload)
    projection.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    if observed != expected:
        raise RepairPackageError(f"{label} self digest drifted.")


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or any(not part for part in path.parts)
    ):
        raise RepairPackageError(f"Unsafe {label}: {value!r}")
    return path


def _exclusive_write(
    path: Path,
    payload: bytes,
    *,
    executable: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RepairPackageError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_write(path, canonical_json_bytes(payload) + b"\n")


def _file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    verify_self_digest(payload, label=f"{path.name} binding")
    return {
        **_file_binding(path, relative_to=relative_to),
        "canonical_sha256": str(payload["sha256"]),
    }


def _parent_context(campaign: Campaign) -> dict[str, Any]:
    package_manifest_path = campaign.parent_package / "package_manifest.json"
    activation_manifest_path = (
        campaign.parent_activation / "activation_manifest.json"
    )
    package_manifest = load_json(
        package_manifest_path, label="parent package manifest"
    )
    activation_manifest = load_json(
        activation_manifest_path, label="parent activation manifest"
    )
    submission = load_json(
        campaign.parent_submission_receipt,
        label="parent submission receipt",
    )
    for label, payload in (
        ("parent package manifest", package_manifest),
        ("parent activation manifest", activation_manifest),
        ("parent submission receipt", submission),
    ):
        verify_self_digest(payload, label=label)
    if (
        int(submission.get("cluster_id", -1))
        != campaign.parent_cluster_id
        or submission.get("source_archive_sha256")
        != package_manifest.get("source_archive", {}).get("sha256")
        or submission.get("package_manifest_sha256")
        != package_manifest.get("sha256")
        or submission.get("activation_manifest_sha256")
        != activation_manifest.get("sha256")
    ):
        raise RepairPackageError(
            f"{campaign.key} parent submission bindings drifted."
        )
    executions = activation_manifest.get("executions")
    if not isinstance(executions, list) or not executions:
        raise RepairPackageError(
            f"{campaign.key} parent activation has no executions."
        )
    if any(
        int(row.get("queue_index", -1)) != index
        for index, row in enumerate(executions)
        if isinstance(row, Mapping)
    ):
        raise RepairPackageError(
            f"{campaign.key} parent queue indices are not contiguous."
        )
    if any(
        proc < 0 or proc >= len(executions)
        for proc in campaign.completed_proc_ids
    ):
        raise RepairPackageError(
            f"{campaign.key} completed proc set is out of range."
        )
    return {
        "package_manifest_path": package_manifest_path,
        "package_manifest": package_manifest,
        "activation_manifest_path": activation_manifest_path,
        "activation_manifest": activation_manifest,
        "submission_path": campaign.parent_submission_receipt,
        "submission": submission,
        "executions": executions,
    }


def _read_parent_source(
    campaign: Campaign,
    context: Mapping[str, Any],
) -> tuple[list[str], dict[str, bytes], dict[str, dict[str, Any]]]:
    manifest_path = campaign.parent_package / "source_archive_manifest.json"
    manifest = load_json(manifest_path, label="parent source manifest")
    verify_self_digest(manifest, label="parent source manifest")
    archive_path = campaign.parent_package / "source_locked.tar.gz"
    archive_binding = manifest.get("archive")
    rows = manifest.get("members")
    if (
        not isinstance(archive_binding, Mapping)
        or not isinstance(rows, list)
        or archive_binding.get("sha256") != sha256_file(archive_path)
        or archive_binding.get("sha256")
        != context["submission"]["source_archive_sha256"]
    ):
        raise RepairPackageError(
            f"{campaign.key} parent source archive binding drifted."
        )
    declared = {
        str(row["path"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if len(declared) != len(rows):
        raise RepairPackageError(
            f"{campaign.key} parent source members duplicate."
        )
    order: list[str] = []
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = _safe_relative(
                member.name, label="parent source archive member"
            ).as_posix()
            if (
                relative not in declared
                or relative in payloads
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise RepairPackageError(
                    f"Unsafe parent source member: {relative}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise RepairPackageError(
                    f"Unreadable parent source member: {relative}"
                )
            payload = stream.read()
            row = declared[relative]
            if (
                sha256_bytes(payload) != row.get("sha256")
                or len(payload) != int(row.get("size_bytes", -1))
            ):
                raise RepairPackageError(
                    f"Parent source member drifted: {relative}"
                )
            row["mode"] = int(member.mode)
            order.append(relative)
            payloads[relative] = payload
    if set(order) != set(declared) or order != sorted(order):
        raise RepairPackageError(
            f"{campaign.key} parent source closure/order drifted."
        )
    if (
        CHECKPOINT_MEMBER not in payloads
        or sha256_bytes(payloads[CHECKPOINT_MEMBER])
        != PARENT_CHECKPOINT_SHA256
    ):
        raise RepairPackageError(
            f"{campaign.key} parent checkpoint source drifted."
        )
    return order, payloads, declared


def _write_source_archive(
    path: Path,
    *,
    order: Sequence[str],
    payloads: Mapping[str, bytes],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or path.is_symlink():
        raise RepairPackageError(f"Refusing to overwrite: {path}")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for relative in order:
                        payload = payloads[relative]
                        info = tarfile.TarInfo(relative)
                        info.size = len(payload)
                        info.mode = int(metadata[relative]["mode"])
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parent_job_path(
    campaign: Campaign,
    execution: Mapping[str, Any],
) -> Path:
    binding = execution.get("job")
    if not isinstance(binding, Mapping):
        raise RepairPackageError("Parent execution job binding is malformed.")
    raw = str(binding.get("path", ""))
    path = REPO_ROOT / _safe_relative(raw, label="parent job path")
    try:
        path.relative_to(campaign.parent_package)
    except ValueError as exc:
        raise RepairPackageError(
            "Parent job escaped its sealed package."
        ) from exc
    return path


def _parent_authorization_path(
    campaign: Campaign,
    execution: Mapping[str, Any],
) -> Path:
    binding = execution.get("authorization")
    if not isinstance(binding, Mapping):
        raise RepairPackageError(
            "Parent execution authorization binding is malformed."
        )
    raw = str(binding.get("path", ""))
    path = campaign.parent_activation / _safe_relative(
        raw, label="parent authorization path"
    )
    return path


def _submission_text(
    campaign: Campaign,
    *,
    source_sha256: str,
    package_relative: Path,
    activation_relative: Path,
) -> str:
    runtime_relative = Path(
        "chtc/paper_i_ra_adapt_repair_20260727/"
        f"{campaign.package_dirname}_runtime"
    )
    return f"""# Frozen, quota-safe v2 checkpoint-retention repair subset.
universe = vanilla
executable = /bin/bash
transfer_executable = False

arguments = {activation_relative.as_posix()}/execute_authorized_job.sh {activation_relative.as_posix()} {package_relative.as_posix()} $(job_path) $(job_file_sha256) $(authorization_path) $(authorization_file_sha256) {source_sha256} {REMOTE_IMAGE_PATH} {REMOTE_IMAGE_SHA256} transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
preserve_relative_paths = True
transfer_input_files = {package_relative.as_posix()}, {activation_relative.as_posix()}, {REMOTE_IMAGE_PATH}
transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz
transfer_output_remaps = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz={runtime_relative.as_posix()}/fetched/$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"

request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
requirements = TARGET.HasSIF

notification = Never
getenv = False
stream_output = False
stream_error = False
+JobBatchName = "{campaign.batch_name}"
max_materialize = {campaign.max_materialize}
max_idle = 0
leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)

log = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).log
output = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).out
error = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).err

queue execution_id,job_path,job_file_sha256,authorization_path,authorization_file_sha256,cpus,memory_mb,disk_mb,max_runtime_seconds from {activation_relative.as_posix()}/queue.tsv
"""


def _ordinary_held_submission_text(
    campaign: Campaign,
    *,
    source_sha256: str,
    package_relative: Path,
    activation_relative: Path,
) -> str:
    runtime_relative = Path(
        "chtc/paper_i_ra_adapt_repair_20260727/"
        f"{campaign.package_dirname}_runtime"
    )
    return f"""# CHTC-compatible ordinary held-start release-v2 activation.
universe = vanilla
executable = /bin/bash
transfer_executable = False

arguments = {activation_relative.as_posix()}/execute_authorized_job.sh {activation_relative.as_posix()} {package_relative.as_posix()} $(job_path) $(job_file_sha256) $(authorization_path) $(authorization_file_sha256) {source_sha256} {REMOTE_IMAGE_PATH} {REMOTE_IMAGE_SHA256} transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
preserve_relative_paths = True
transfer_input_files = {package_relative.as_posix()}, {activation_relative.as_posix()}, {REMOTE_IMAGE_PATH}
transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz
transfer_output_remaps = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz={runtime_relative.as_posix()}/fetched/$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz"

request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
requirements = TARGET.HasSIF

notification = Never
getenv = False
stream_output = False
stream_error = False
+JobBatchName = "{campaign.ordinary_held_release_batch_name}"
+HolsteinLifecycleMode = "ordinary_held_exact_proc_release_v1"
hold = True
periodic_release = False
leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)

log = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).log
output = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).out
error = {runtime_relative.as_posix()}/logs/$(Cluster).$(Process)__$(execution_id).err

queue execution_id,job_path,job_file_sha256,authorization_path,authorization_file_sha256,cpus,memory_mb,disk_mb,max_runtime_seconds from {activation_relative.as_posix()}/queue.tsv
"""


def _verified_completion_from_rolling_observation(
    campaign: Campaign,
    *,
    context: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_id: str,
    proc_id: int,
) -> dict[str, Any]:
    receipt_path = REPO_ROOT / ROLLING_RECEIPT_RELATIVE
    receipt = load_json(receipt_path, label="rolling retrieval receipt")
    verify_self_digest(receipt, label="rolling retrieval receipt")
    candidates = [
        row
        for group in ("remote_preserved_archives", "local_unpaired_archives")
        for row in receipt.get(group, [])
        if isinstance(row, Mapping)
        and int(row.get("cluster_id", -1)) == campaign.parent_cluster_id
        and int(row.get("proc_id", -1)) == proc_id
    ]
    if not candidates:
        raise RepairPackageError(
            f"No completion evidence for {campaign.key} proc {proc_id}."
        )
    row = dict(candidates[0])
    row_execution_id = row.get("execution_id")
    if row_execution_id not in {None, execution_id}:
        raise RepairPackageError(
            f"Completion evidence execution mismatch for proc {proc_id}."
        )
    local_relative = _safe_relative(
        str(row.get("local_path", "")),
        label="completed v1 local archive",
    )
    local_path = REPO_ROOT / local_relative
    if (
        not local_path.is_file()
        or local_path.is_symlink()
        or local_path.stat().st_size
        != int(row.get("size_bytes", -1))
    ):
        raise RepairPackageError(
            f"Completed v1 archive is not locally preserved: {local_path}"
        )
    expected_archive_sha256 = row.get("local_sha256")
    if not isinstance(expected_archive_sha256, str):
        expected_archive_sha256 = row.get("remote_sha256")
    observed_archive_sha256 = sha256_file(local_path)
    if (
        not isinstance(expected_archive_sha256, str)
        or observed_archive_sha256 != expected_archive_sha256
    ):
        raise RepairPackageError(
            f"Completed v1 archive digest drifted: {local_path}"
        )

    member_names: list[str] = []
    attempt_receipt: dict[str, Any] | None = None
    with tarfile.open(local_path, "r|gz") as archive:
        for member in archive:
            member_name = _safe_relative(
                member.name, label="completed v1 archive member"
            ).as_posix()
            if (
                member_name in member_names
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise RepairPackageError(
                    f"Unsafe completed v1 archive member: {member_name}"
                )
            member_names.append(member_name)
            if member_name == "worker_attempt_receipt.json":
                stream = archive.extractfile(member)
                if stream is None:
                    raise RepairPackageError(
                        "Unreadable worker attempt receipt."
                    )
                try:
                    parsed = json.loads(stream.read())
                except json.JSONDecodeError as exc:
                    raise RepairPackageError(
                        "Malformed worker attempt receipt."
                    ) from exc
                if not isinstance(parsed, dict):
                    raise RepairPackageError(
                        "Worker attempt receipt is not an object."
                    )
                attempt_receipt = parsed
    if attempt_receipt is None:
        raise RepairPackageError(
            f"Completed v1 archive has no attempt receipt: {local_path}"
        )
    verify_self_digest(
        attempt_receipt, label="completed v1 worker attempt receipt"
    )
    worker_rows = attempt_receipt.get("worker_files")
    if not isinstance(worker_rows, list):
        raise RepairPackageError(
            "Completed v1 worker file inventory is malformed."
        )
    worker_paths = {
        str(worker_row.get("path"))
        for worker_row in worker_rows
        if isinstance(worker_row, Mapping)
        and isinstance(worker_row.get("path"), str)
    }
    expected_members = {
        f"worker_outputs/{worker_path}" for worker_path in worker_paths
    } | {
        "authority/job.json",
        "authority/execution_authorization.json",
        "authority/activation_manifest.json",
        "worker_attempt_receipt.json",
    }
    parent_job_path = _parent_job_path(campaign, execution)
    parent_authorization_path = _parent_authorization_path(
        campaign, execution
    )
    if (
        len(worker_paths) != len(worker_rows)
        or set(member_names) != expected_members
        or "attempt_identity.tsv" not in worker_paths
        or "result.json" not in worker_paths
        or "worker_exit_status.txt" not in worker_paths
        or attempt_receipt.get("execution_id") != execution_id
        or int(attempt_receipt.get("cluster_id", -1))
        != campaign.parent_cluster_id
        or int(attempt_receipt.get("proc_id", -1)) != proc_id
        or int(attempt_receipt.get("attempt_ordinal", -1)) < 1
        or int(attempt_receipt.get("worker_exit_status", -1)) != 0
        or attempt_receipt.get("job_file_sha256")
        != sha256_file(parent_job_path)
        or attempt_receipt.get("authorization_file_sha256")
        != sha256_file(parent_authorization_path)
        or attempt_receipt.get("activation_manifest_file_sha256")
        != sha256_file(context["activation_manifest_path"])
        or attempt_receipt.get("source_archive_sha256")
        != context["submission"]["source_archive_sha256"]
        or attempt_receipt.get("image_sha256")
        != REMOTE_IMAGE_SHA256
    ):
        raise RepairPackageError(
            f"Completed v1 archive authority closure failed: {local_path}"
        )
    return {
        "rolling_receipt": {
            "path": ROLLING_RECEIPT_RELATIVE.as_posix(),
            "file_sha256": sha256_file(receipt_path),
            "canonical_sha256": receipt["sha256"],
        },
        "archive_observation": row,
        "local_verification": {
            "path": local_relative.as_posix(),
            "sha256": observed_archive_sha256,
            "size_bytes": local_path.stat().st_size,
            "gzip_and_full_tar_scan_passed": True,
            "regular_member_closure_passed": True,
            "member_count": len(member_names),
            "worker_attempt_receipt_sha256": attempt_receipt["sha256"],
            "worker_exit_status": 0,
            "authority_bindings_passed": True,
            "status": "passed",
        },
    }


def seal_preservation_snapshot(
    *,
    output_path: Path,
    sealed_utc: str,
) -> dict[str, Any]:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RepairPackageError(
            "Preservation snapshot must be written under the repository."
        ) from exc
    rows: list[dict[str, Any]] = []
    campaign_bindings: list[dict[str, Any]] = []
    for campaign in CAMPAIGNS:
        context = _parent_context(campaign)
        campaign_bindings.append(
            {
                "campaign_key": campaign.key,
                "cluster_id": campaign.parent_cluster_id,
                "parent_package": _json_binding(
                    context["package_manifest_path"],
                    relative_to=REPO_ROOT,
                ),
                "parent_activation": _json_binding(
                    context["activation_manifest_path"],
                    relative_to=REPO_ROOT,
                ),
                "parent_submission_receipt": _json_binding(
                    context["submission_path"],
                    relative_to=REPO_ROOT,
                ),
            }
        )
        for proc_id in campaign.completed_proc_ids:
            execution = context["executions"][proc_id]
            execution_id = str(execution["execution_id"])
            evidence = _verified_completion_from_rolling_observation(
                campaign,
                context=context,
                execution=execution,
                execution_id=execution_id,
                proc_id=proc_id,
            )
            observation = evidence["archive_observation"]
            verification = evidence["local_verification"]
            remote_disposition = {
                key: observation[key]
                for key in (
                    "remote_path",
                    "remote_sha256",
                    "remote_copy_observed",
                    "remote_pathname_state",
                    "remote_pathname_absent_observed_utc",
                    "agent_delete_observed",
                    "remote_delete_attribution",
                )
                if key in observation
            }
            rows.append(
                digested(
                    {
                        "campaign_key": campaign.key,
                        "cluster_id": campaign.parent_cluster_id,
                        "proc_id": proc_id,
                        "execution_id": execution_id,
                        "archive": {
                            "path": verification["path"],
                            "sha256": verification["sha256"],
                            "size_bytes": verification["size_bytes"],
                        },
                        "local_verification": {
                            key: value
                            for key, value in verification.items()
                            if key not in {"path", "sha256", "size_bytes"}
                        },
                        "remote_disposition": remote_disposition,
                        "scientific_result_disposition": (
                            "completed_v1_preserved_excluded_from_v2"
                        ),
                        "status": "passed",
                    }
                )
            )
    snapshot = digested(
        {
            "schema": PRESERVATION_SNAPSHOT_SCHEMA,
            "sealed_utc": sealed_utc,
            "source_observation_class": (
                "mutable_rolling_guard_receipt_observed_but_not_bound"
            ),
            "rolling_guard_receipt_bound": False,
            "campaigns": campaign_bindings,
            "completed_archive_count": len(rows),
            "completed_archive_total_bytes": sum(
                int(row["archive"]["size_bytes"]) for row in rows
            ),
            "rows": rows,
            "all_archive_size_sha256_gzip_tar_authority_checks_passed": True,
            "unrelated_clusters_touched": False,
            "status": "passed",
        }
    )
    _write_json(output, snapshot)
    return snapshot


def validate_preservation_snapshot(path: Path) -> dict[str, Any]:
    snapshot_path = path.resolve()
    try:
        snapshot_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RepairPackageError(
            "Preservation snapshot escaped the repository."
        ) from exc
    snapshot = load_json(
        snapshot_path, label="completed v1 preservation snapshot"
    )
    verify_self_digest(
        snapshot, label="completed v1 preservation snapshot"
    )
    expected_keys = [
        (campaign.key, campaign.parent_cluster_id, proc_id)
        for campaign in CAMPAIGNS
        for proc_id in campaign.completed_proc_ids
    ]
    expected_campaign_bindings = []
    for campaign in CAMPAIGNS:
        context = _parent_context(campaign)
        expected_campaign_bindings.append(
            {
                "campaign_key": campaign.key,
                "cluster_id": campaign.parent_cluster_id,
                "parent_package": _json_binding(
                    context["package_manifest_path"],
                    relative_to=REPO_ROOT,
                ),
                "parent_activation": _json_binding(
                    context["activation_manifest_path"],
                    relative_to=REPO_ROOT,
                ),
                "parent_submission_receipt": _json_binding(
                    context["submission_path"],
                    relative_to=REPO_ROOT,
                ),
            }
        )
    rows = snapshot.get("rows")
    if (
        snapshot.get("schema") != PRESERVATION_SNAPSHOT_SCHEMA
        or not snapshot.get("sealed_utc")
        or snapshot.get("rolling_guard_receipt_bound") is not False
        or snapshot.get("campaigns") != expected_campaign_bindings
        or snapshot.get("completed_archive_count") != len(expected_keys)
        or not isinstance(rows, list)
        or [
            (
                row.get("campaign_key"),
                int(row.get("cluster_id", -1)),
                int(row.get("proc_id", -1)),
            )
            for row in rows
            if isinstance(row, Mapping)
        ]
        != expected_keys
        or snapshot.get(
            "all_archive_size_sha256_gzip_tar_authority_checks_passed"
        )
        is not True
        or snapshot.get("unrelated_clusters_touched") is not False
        or snapshot.get("status") != "passed"
    ):
        raise RepairPackageError(
            "Completed v1 preservation snapshot did not close."
        )
    total_bytes = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise RepairPackageError(
                "Preservation snapshot row is malformed."
            )
        verify_self_digest(row, label="preservation snapshot row")
        archive = row.get("archive")
        verification = row.get("local_verification")
        if (
            not isinstance(archive, Mapping)
            or not isinstance(verification, Mapping)
            or len(str(archive.get("sha256", ""))) != 64
            or int(archive.get("size_bytes", -1)) <= 0
            or verification.get("gzip_and_full_tar_scan_passed")
            is not True
            or verification.get("authority_bindings_passed") is not True
            or int(verification.get("worker_exit_status", -1)) != 0
            or verification.get("status") != "passed"
            or row.get("status") != "passed"
        ):
            raise RepairPackageError(
                "Preservation snapshot verification row drifted."
            )
        total_bytes += int(archive["size_bytes"])
    if total_bytes != int(
        snapshot.get("completed_archive_total_bytes", -1)
    ):
        raise RepairPackageError(
            "Preservation snapshot byte closure drifted."
        )
    return snapshot


def _completion_evidence(
    campaign: Campaign,
    *,
    context: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_id: str,
    proc_id: int,
) -> dict[str, Any]:
    del context, execution
    snapshot_path = REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE
    snapshot = validate_preservation_snapshot(snapshot_path)
    matches = [
        row
        for row in snapshot["rows"]
        if row["campaign_key"] == campaign.key
        and int(row["cluster_id"]) == campaign.parent_cluster_id
        and int(row["proc_id"]) == proc_id
        and row["execution_id"] == execution_id
    ]
    if len(matches) != 1:
        raise RepairPackageError(
            f"Stable completion snapshot mismatch for "
            f"{campaign.key} proc {proc_id}."
        )
    return {
        "preservation_snapshot": _json_binding(
            snapshot_path, relative_to=REPO_ROOT
        ),
        "snapshot_row": matches[0],
    }


def _repo_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RepairPackageError(
            f"{label} must be preserved under the active repository."
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise RepairPackageError(f"Missing or unsafe {label}: {resolved}")
    return resolved


def seal_retirement_receipt(
    *,
    output_path: Path,
    retired_utc: str,
    verified_utc: str,
    schedd: str,
    owner: str,
    condor_rm_stdout: Path,
    post_condor_q_json: Path,
    post_condor_q_factory_json: Path,
    condor_rm_exit_status: int,
    post_condor_q_exit_status: int,
    post_condor_q_factory_exit_status: int,
) -> dict[str, Any]:
    output = output_path.resolve()
    try:
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RepairPackageError(
            "Retirement receipt must be written under the repository."
        ) from exc
    removal = _repo_file(
        condor_rm_stdout, label="condor_rm stdout evidence"
    )
    live_query = _repo_file(
        post_condor_q_json, label="post-retirement condor_q JSON"
    )
    factory_query = _repo_file(
        post_condor_q_factory_json,
        label="post-retirement factory condor_q JSON",
    )
    try:
        live_ads = json.loads(live_query.read_text(encoding="utf-8"))
        factory_ads = json.loads(factory_query.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RepairPackageError(
            "Post-retirement Condor evidence is not JSON."
        ) from exc
    if live_ads != [] or factory_ads != []:
        raise RepairPackageError(
            "Parent clusters still have live or factory ads."
        )
    if (
        condor_rm_exit_status != 0
        or post_condor_q_exit_status != 0
        or post_condor_q_factory_exit_status != 0
    ):
        raise RepairPackageError(
            "Exact retirement or post-retirement query failed."
        )
    cluster_ids = [campaign.parent_cluster_id for campaign in CAMPAIGNS]
    receipt = digested(
        {
            "schema": RETIREMENT_RECEIPT_SCHEMA,
            "target_cluster_ids": cluster_ids,
            "retirement_command": (
                "condor_rm " + " ".join(str(value) for value in cluster_ids)
            ),
            "retirement_command_scope": "exact_target_clusters_only",
            "condor_rm_exit_status": condor_rm_exit_status,
            "post_condor_q_exit_status": post_condor_q_exit_status,
            "post_condor_q_factory_exit_status": (
                post_condor_q_factory_exit_status
            ),
            "removed_cluster_ids": cluster_ids,
            "retired_utc": retired_utc,
            "post_retirement_verified_utc": verified_utc,
            "schedd": schedd,
            "owner": owner,
            "evidence": {
                "condor_rm_stdout": _file_binding(
                    removal, relative_to=REPO_ROOT
                ),
                "post_condor_q_json": _file_binding(
                    live_query, relative_to=REPO_ROOT
                ),
                "post_condor_q_factory_json": _file_binding(
                    factory_query, relative_to=REPO_ROOT
                ),
            },
            "post_retirement_live_ads": live_ads,
            "post_retirement_factory_ads": factory_ads,
            "post_retirement_live_ad_count": 0,
            "post_retirement_factory_ad_count": 0,
            "all_target_clusters_absent_from_queue": True,
            "unrelated_clusters_touched": False,
            "status": "passed",
        }
    )
    _write_json(output, receipt)
    return receipt


def validate_retirement_receipt(path: Path) -> dict[str, Any]:
    receipt_path = _repo_file(path, label="retirement receipt")
    receipt = load_json(
        receipt_path, label="parent cluster retirement receipt"
    )
    verify_self_digest(
        receipt, label="parent cluster retirement receipt"
    )
    cluster_ids = [campaign.parent_cluster_id for campaign in CAMPAIGNS]
    if (
        receipt.get("schema") != RETIREMENT_RECEIPT_SCHEMA
        or receipt.get("target_cluster_ids") != cluster_ids
        or receipt.get("removed_cluster_ids") != cluster_ids
        or receipt.get("retirement_command")
        != "condor_rm " + " ".join(str(value) for value in cluster_ids)
        or receipt.get("retirement_command_scope")
        != "exact_target_clusters_only"
        or int(receipt.get("condor_rm_exit_status", -1)) != 0
        or int(receipt.get("post_condor_q_exit_status", -1)) != 0
        or int(
            receipt.get("post_condor_q_factory_exit_status", -1)
        )
        != 0
        or not receipt.get("retired_utc")
        or not receipt.get("post_retirement_verified_utc")
        or not receipt.get("schedd")
        or not receipt.get("owner")
        or receipt.get("post_retirement_live_ads") != []
        or receipt.get("post_retirement_factory_ads") != []
        or int(receipt.get("post_retirement_live_ad_count", -1)) != 0
        or int(receipt.get("post_retirement_factory_ad_count", -1)) != 0
        or receipt.get("all_target_clusters_absent_from_queue") is not True
        or receipt.get("unrelated_clusters_touched") is not False
        or receipt.get("status") != "passed"
    ):
        raise RepairPackageError(
            "Parent cluster retirement receipt did not close."
        )
    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RepairPackageError("Retirement evidence bindings are absent.")
    expected_json: dict[str, Any] = {
        "post_condor_q_json": [],
        "post_condor_q_factory_json": [],
    }
    for label in (
        "condor_rm_stdout",
        "post_condor_q_json",
        "post_condor_q_factory_json",
    ):
        binding = evidence.get(label)
        if not isinstance(binding, Mapping):
            raise RepairPackageError(
                f"Retirement evidence binding is absent: {label}"
            )
        evidence_path = REPO_ROOT / _safe_relative(
            str(binding.get("path", "")),
            label=f"{label} evidence path",
        )
        if (
            not evidence_path.is_file()
            or evidence_path.is_symlink()
            or sha256_file(evidence_path) != binding.get("sha256")
            or evidence_path.stat().st_size
            != int(binding.get("size_bytes", -1))
        ):
            raise RepairPackageError(
                f"Retirement evidence file drifted: {label}"
            )
        if label in expected_json:
            try:
                observed = json.loads(
                    evidence_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise RepairPackageError(
                    f"Retirement evidence JSON drifted: {label}"
                ) from exc
            if observed != expected_json[label]:
                raise RepairPackageError(
                    f"Retirement evidence is no longer empty: {label}"
                )
    return receipt


def materialize_campaign(
    campaign: Campaign,
    *,
    output_root: Path,
    authorized_utc: str,
) -> dict[str, Path]:
    context = _parent_context(campaign)
    preservation_snapshot_path = (
        REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE
    )
    validate_preservation_snapshot(preservation_snapshot_path)
    preservation_snapshot_binding = _json_binding(
        preservation_snapshot_path, relative_to=REPO_ROOT
    )
    package_dir = output_root / campaign.package_dirname
    activation_dir = output_root / campaign.activation_dirname
    if package_dir.exists() or activation_dir.exists():
        raise RepairPackageError(
            f"Refusing to overwrite materialized v2 {campaign.key} paths."
        )
    package_dir.mkdir(parents=True)
    activation_dir.mkdir(parents=True)

    order, parent_payloads, parent_rows = _read_parent_source(
        campaign, context
    )
    repaired_checkpoint = (
        REPO_ROOT / CHECKPOINT_MEMBER
    ).read_bytes()
    if sha256_bytes(repaired_checkpoint) != REPAIRED_CHECKPOINT_SHA256:
        raise RepairPackageError(
            "Approved current-checkpoint repair hash drifted."
        )
    repaired_payloads = dict(parent_payloads)
    repaired_payloads[CHECKPOINT_MEMBER] = repaired_checkpoint
    archive_path = package_dir / "source_locked.tar.gz"
    _write_source_archive(
        archive_path,
        order=order,
        payloads=repaired_payloads,
        metadata=parent_rows,
    )
    archive_sha256 = sha256_file(archive_path)
    if (
        campaign.expected_source_archive_sha256 is not None
        and archive_sha256 != campaign.expected_source_archive_sha256
    ):
        raise RepairPackageError(
            f"{campaign.key} deterministic source archive drifted."
        )
    member_rows = [
        {
            "path": relative,
            "sha256": sha256_bytes(repaired_payloads[relative]),
            "size_bytes": len(repaired_payloads[relative]),
            "mode": int(parent_rows[relative]["mode"]),
        }
        for relative in order
    ]
    source_manifest = digested(
        {
            "schema": (
                "paper_i_checkpoint_sidecar_retention_source_archive_v2"
            ),
            "operational_package_id": campaign.operational_package_id,
            "scientific_parent_package_id": context[
                "package_manifest"
            ]["package_id"],
            "status": "passed",
            "archive": {
                "path": "source_locked.tar.gz",
                "sha256": archive_sha256,
                "size_bytes": archive_path.stat().st_size,
            },
            "parent_archive": {
                "path": (
                    campaign.parent_package
                    / "source_locked.tar.gz"
                ).relative_to(REPO_ROOT).as_posix(),
                "sha256": context["submission"][
                    "source_archive_sha256"
                ],
            },
            "member_count": len(member_rows),
            "members": member_rows,
            "deterministic_archive": {
                "gzip_mtime": 0,
                "tar_member_mtime": 0,
                "uid": 0,
                "gid": 0,
                "ordered_by_path": True,
                "parent_member_modes_preserved": True,
            },
        }
    )
    source_manifest_path = package_dir / "source_archive_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    unchanged_count = sum(
        repaired_payloads[name] == parent_payloads[name]
        for name in order
    )
    delta = digested(
        {
            "schema": (
                "paper_i_checkpoint_sidecar_retention_source_delta_v2"
            ),
            "operational_package_id": campaign.operational_package_id,
            "scientific_parent_package_id": context[
                "package_manifest"
            ]["package_id"],
            "parent_source_archive_sha256": context["submission"][
                "source_archive_sha256"
            ],
            "repaired_source_archive_sha256": archive_sha256,
            "parent_member_count": len(order),
            "repaired_member_count": len(order),
            "unchanged_member_count": unchanged_count,
            "changed_member_count": 1,
            "changed_members": [
                {
                    "path": CHECKPOINT_MEMBER,
                    "parent_sha256": PARENT_CHECKPOINT_SHA256,
                    "parent_size_bytes": len(
                        parent_payloads[CHECKPOINT_MEMBER]
                    ),
                    "parent_mode": int(
                        parent_rows[CHECKPOINT_MEMBER]["mode"]
                    ),
                    "repaired_sha256": REPAIRED_CHECKPOINT_SHA256,
                    "repaired_size_bytes": len(repaired_checkpoint),
                    "repaired_mode": int(
                        parent_rows[CHECKPOINT_MEMBER]["mode"]
                    ),
                    "classification": (
                        "observation_only_authenticated_predecessor_"
                        "sidecar_retirement"
                    ),
                    "scientific_protocol_change": False,
                    "controller_semantics_change": False,
                }
            ],
            "protocol_members_byte_identical": True,
            "non_checkpoint_members_byte_identical": True,
            "ordered_member_paths_identical": True,
            "member_modes_identical": True,
            "scientific_settings_changed": [],
            "status": "passed",
        }
    )
    delta_path = package_dir / "source_delta_receipt.json"
    _write_json(delta_path, delta)

    for name in ("run_cell.py", "package_contract.py"):
        source = campaign.parent_package / name
        _exclusive_write(
            package_dir / name,
            source.read_bytes(),
            executable=bool(source.stat().st_mode & 0o111),
        )

    executions = context["executions"]
    completed = set(campaign.completed_proc_ids)
    remaining = [
        (proc_id, execution)
        for proc_id, execution in enumerate(executions)
        if proc_id not in completed
    ]
    jobs_dir = package_dir / "jobs"
    jobs_dir.mkdir()
    job_rows: list[dict[str, Any]] = []
    plan_ids: list[str] = []
    for queue_index, (parent_proc_id, execution) in enumerate(remaining):
        if not isinstance(execution, Mapping):
            raise RepairPackageError("Parent execution row is malformed.")
        execution_id = str(execution["execution_id"])
        source = _parent_job_path(campaign, execution)
        destination = jobs_dir / f"{execution_id}.json"
        _exclusive_write(destination, source.read_bytes())
        if sha256_file(destination) != sha256_file(source):
            raise RepairPackageError("Parent job byte copy drifted.")
        job = load_json(destination, label="copied parent job")
        verify_self_digest(job, label="copied parent job")
        plan_ids.append(execution_id)
        job_rows.append(
            {
                "queue_index": queue_index,
                "parent_proc_id": parent_proc_id,
                "execution_id": execution_id,
                "job": _json_binding(
                    destination, relative_to=package_dir
                ),
                "parent_job": _json_binding(
                    source, relative_to=REPO_ROOT
                ),
                "protocol": dict(job["protocol"]),
                "scientific_job_bytes_identical": True,
            }
        )

    supersession_rows: list[dict[str, Any]] = []
    queue_index_by_parent = {
        parent_proc: queue_index
        for queue_index, (parent_proc, _) in enumerate(remaining)
    }
    for parent_proc_id, execution in enumerate(executions):
        execution_id = str(execution["execution_id"])
        if parent_proc_id in completed:
            supersession_rows.append(
                {
                    "parent_cluster_id": campaign.parent_cluster_id,
                    "parent_proc_id": parent_proc_id,
                    "execution_id": execution_id,
                    "state": "completed_v1_preserved_excluded_from_v2",
                    "v2_queue_index": None,
                    "completion_evidence": _completion_evidence(
                        campaign,
                        context=context,
                        execution=execution,
                        execution_id=execution_id,
                        proc_id=parent_proc_id,
                    ),
                }
            )
        else:
            supersession_rows.append(
                {
                    "parent_cluster_id": campaign.parent_cluster_id,
                    "parent_proc_id": parent_proc_id,
                    "execution_id": execution_id,
                    "state": "uncompleted_v1_superseded_by_v2",
                    "v2_queue_index": queue_index_by_parent[
                        parent_proc_id
                    ],
                }
            )
    supersession = digested(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_supersession_map_v2"
            ),
            "campaign_key": campaign.key,
            "campaign_id": context["package_manifest"]["campaign_id"],
            "parent_cluster_id": campaign.parent_cluster_id,
            "parent_package": _json_binding(
                context["package_manifest_path"],
                relative_to=REPO_ROOT,
            ),
            "parent_activation": _json_binding(
                context["activation_manifest_path"],
                relative_to=REPO_ROOT,
            ),
            "parent_submission_receipt": _json_binding(
                context["submission_path"],
                relative_to=REPO_ROOT,
            ),
            "operational_package_id": campaign.operational_package_id,
            "activation_id": campaign.activation_id,
            "completed_v1_preservation_snapshot": (
                preservation_snapshot_binding
            ),
            "parent_row_count": len(executions),
            "campaign_cell_count": campaign.campaign_cell_count,
            "completed_v1_excluded_count": len(completed),
            "v2_queued_count": len(remaining),
            "completed_parent_proc_ids": sorted(completed),
            "rows": supersession_rows,
            "unrelated_clusters_touched": False,
            "status": "passed",
        }
    )
    supersession_path = package_dir / "supersession_map.json"
    _write_json(supersession_path, supersession)

    plan = digested(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_execution_plan_v2"
            ),
            "operational_package_id": campaign.operational_package_id,
            "scientific_parent_package_id": context[
                "package_manifest"
            ]["package_id"],
            "campaign_id": context["package_manifest"]["campaign_id"],
            "run_class": context["package_manifest"]["run_class"],
            "repair_class": "implementation_plumbing_observation_retention",
            "execution_target": "chtc",
            "execution_count": len(remaining),
            "campaign_cell_count": campaign.campaign_cell_count,
            "direct_execution_count": len(remaining),
            "completed_predecessor_count": len(completed),
            "execution_ids": plan_ids,
            "parent_proc_ids": [proc for proc, _ in remaining],
            "completed_v1_excluded_proc_ids": sorted(completed),
            "source_archive_sha256": archive_sha256,
            "source_delta_sha256": delta["sha256"],
            "supersession_map_sha256": supersession["sha256"],
            "completed_v1_preservation_snapshot": (
                preservation_snapshot_binding
            ),
            "scientific_protocols_reused_byte_for_byte": True,
            "scientific_job_contracts_reused_byte_for_byte": True,
            "scientific_settings_changed": [],
            "completed_predecessor_local_preservation_verified": True,
            "parent_cluster_retirement_verified": False,
            "execution_thaw_authorized": False,
            "execution_thaw_blockers": [
                "parent_cluster_retirement_not_verified_by_local_builder"
            ],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        }
    )
    plan_path = package_dir / "execution_plan.json"
    _write_json(plan_path, plan)

    package_queue_lines = [
        "\t".join(
            (
                row["execution_id"],
                str(row["parent_proc_id"]),
                str(row["queue_index"]),
            )
        )
        + "\n"
        for row in job_rows
    ]
    package_queue_path = package_dir / "queue.tsv"
    _exclusive_write(
        package_queue_path, "".join(package_queue_lines).encode("utf-8")
    )
    package_manifest = digested(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_package_v2"
            ),
            "operational_package_id": campaign.operational_package_id,
            "scientific_parent_package_id": context[
                "package_manifest"
            ]["package_id"],
            "campaign_id": context["package_manifest"]["campaign_id"],
            "run_class": context["package_manifest"]["run_class"],
            "status": "passed",
            "repair_class": "implementation_plumbing_observation_retention",
            "source_archive": {
                **_file_binding(
                    archive_path, relative_to=package_dir
                ),
                "manifest": _json_binding(
                    source_manifest_path, relative_to=package_dir
                ),
            },
            "source_delta_receipt": _json_binding(
                delta_path, relative_to=package_dir
            ),
            "execution_plan": _json_binding(
                plan_path, relative_to=package_dir
            ),
            "supersession_map": _json_binding(
                supersession_path, relative_to=package_dir
            ),
            "completed_v1_preservation_snapshot": (
                preservation_snapshot_binding
            ),
            "jobs": job_rows,
            "queue": _file_binding(
                package_queue_path, relative_to=package_dir
            ),
            "runtime_control": [
                _file_binding(
                    package_dir / name, relative_to=package_dir
                )
                for name in ("run_cell.py", "package_contract.py")
            ],
            "parent_source_archive_sha256": context["submission"][
                "source_archive_sha256"
            ],
            "campaign_cell_count": campaign.campaign_cell_count,
            "completed_v1_excluded_count": len(completed),
            "completed_predecessor_count": len(completed),
            "direct_execution_count": len(remaining),
            "protocol_files_byte_identical": True,
            "job_files_byte_identical": True,
            "non_checkpoint_source_members_byte_identical": True,
            "scientific_settings_changed": [],
            "completed_predecessor_local_preservation_verified": True,
            "parent_cluster_retirement_verified": False,
            "execution_thaw_authorized": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    package_manifest_path = package_dir / "package_manifest.json"
    _write_json(package_manifest_path, package_manifest)

    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        source = campaign.parent_activation / name
        _exclusive_write(
            activation_dir / name,
            source.read_bytes(),
            executable=bool(source.stat().st_mode & 0o111),
        )
    package_relative = package_dir.relative_to(REPO_ROOT)
    activation_relative = activation_dir.relative_to(REPO_ROOT)
    submit_path = activation_dir / "submit.sub"
    _exclusive_write(
        submit_path,
        _submission_text(
            campaign,
            source_sha256=archive_sha256,
            package_relative=package_relative,
            activation_relative=activation_relative,
        ).encode("utf-8"),
    )
    controls = [
        _file_binding(activation_dir / name, relative_to=activation_dir)
        for name in (
            "execute_authorized_job.sh",
            "build_attempt_archive.py",
            "submit.sub",
        )
    ]
    control_sha256 = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()

    authorizations_dir = activation_dir / "authorizations"
    authorizations_dir.mkdir()
    activation_rows: list[dict[str, Any]] = []
    activation_queue_lines: list[str] = []
    for job_row, (_, parent_execution) in zip(
        job_rows, remaining, strict=True
    ):
        execution_id = job_row["execution_id"]
        job_path = package_dir / job_row["job"]["path"]
        parent_authorization_path = _parent_authorization_path(
            campaign, parent_execution
        )
        parent_authorization = load_json(
            parent_authorization_path,
            label="parent execution authorization",
        )
        verify_self_digest(
            parent_authorization,
            label="parent execution authorization",
        )
        authorization_projection = dict(parent_authorization)
        authorization_projection.pop("sha256", None)
        authorization_projection.update(
            {
                "authorization_id": (
                    f"{campaign.activation_id}__{execution_id}"
                ),
                "authorized_utc": authorized_utc,
                "activation_id": campaign.activation_id,
                "batch_name": campaign.batch_name,
                "source_archive_sha256": archive_sha256,
                "activation_control_plane_sha256": control_sha256,
                "operational_package_id": (
                    campaign.operational_package_id
                ),
                "operational_package_manifest_sha256": (
                    package_manifest["sha256"]
                ),
                "operational_execution_plan_sha256": plan["sha256"],
                "checkpoint_retention_source_delta_sha256": (
                    delta["sha256"]
                ),
                "supersession_map_sha256": supersession["sha256"],
                "completed_v1_preservation_snapshot": (
                    preservation_snapshot_binding
                ),
                "parent_authorization": _json_binding(
                    parent_authorization_path,
                    relative_to=REPO_ROOT,
                ),
                "scientific_parent_job_bytes_identical": True,
                "scientific_settings_changed": [],
                "parent_cluster_retirement_required_before_thaw": True,
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_not_submitted",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
            }
        )
        authorization = digested(authorization_projection)
        authorization_path = (
            authorizations_dir / f"{execution_id}.json"
        )
        _write_json(authorization_path, authorization)
        resources = load_json(
            job_path, label="copied parent job resources"
        )["resources"]
        activation_row = {
            "queue_index": job_row["queue_index"],
            "parent_proc_id": job_row["parent_proc_id"],
            "execution_id": execution_id,
            "job": _json_binding(job_path, relative_to=REPO_ROOT),
            "authorization": _json_binding(
                authorization_path, relative_to=REPO_ROOT
            ),
            "resources": dict(resources),
        }
        activation_rows.append(activation_row)
        activation_queue_lines.append(
            "\t".join(
                (
                    execution_id,
                    job_path.relative_to(REPO_ROOT).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    sha256_file(authorization_path),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )
    activation_queue_path = activation_dir / "queue.tsv"
    _exclusive_write(
        activation_queue_path,
        "".join(activation_queue_lines).encode("utf-8"),
    )
    activation_manifest = digested(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_activation_v2"
            ),
            "activation_id": campaign.activation_id,
            "operational_package_id": campaign.operational_package_id,
            "scientific_parent_package_id": context[
                "package_manifest"
            ]["package_id"],
            "campaign_id": context["package_manifest"]["campaign_id"],
            "batch_name": campaign.batch_name,
            "authorized_utc": authorized_utc,
            "execution_target": "chtc",
            "campaign_cell_count": campaign.campaign_cell_count,
            "direct_execution_count": len(activation_rows),
            "sealed_operational_package": _json_binding(
                package_manifest_path, relative_to=REPO_ROOT
            ),
            "source_archive_sha256": archive_sha256,
            "parent_source_archive_sha256": context["submission"][
                "source_archive_sha256"
            ],
            "source_delta_receipt": _json_binding(
                delta_path, relative_to=REPO_ROOT
            ),
            "supersession_map": _json_binding(
                supersession_path, relative_to=REPO_ROOT
            ),
            "completed_v1_preservation_snapshot": (
                preservation_snapshot_binding
            ),
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "parent_verification_reused": True,
            },
            "control_plane": controls,
            "activation_control_plane_sha256": control_sha256,
            "executions": activation_rows,
            "queue": _file_binding(
                activation_queue_path, relative_to=activation_dir
            ),
            "queue_variables": [
                "execution_id",
                "job_path",
                "job_file_sha256",
                "authorization_path",
                "authorization_file_sha256",
                "cpus",
                "memory_mb",
                "disk_mb",
                "max_runtime_seconds",
            ],
            "factory_contract": {
                "max_materialize": campaign.max_materialize,
                "max_idle": 0,
                "initial_state": "frozen_no_rows_materialized",
                "leave_in_queue": (
                    "(JobStatus == 4) && (ExitCode =!= 0)"
                ),
                "successful_rows_reap": True,
                "failed_rows_retained": True,
                "one_at_a_time_thaw_required": True,
            },
            "pre_execution_gates": {
                "completed_predecessor_local_preservation_verified": True,
                "parent_cluster_retirement_verified": False,
                "execution_thaw_authorized": False,
                "execution_thaw_blockers": [
                    "parent_cluster_retirement_not_verified_by_local_builder"
                ],
                "submission_must_remain_frozen_until_gates_pass": True,
            },
            "scientific_job_bytes_identical": True,
            "scientific_protocol_bytes_identical": True,
            "scientific_settings_changed": [],
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
            "paper_evidence_adopted": False,
        }
    )
    activation_manifest_path = (
        activation_dir / "activation_manifest.json"
    )
    _write_json(activation_manifest_path, activation_manifest)
    return {
        "package_dir": package_dir,
        "activation_dir": activation_dir,
        "package_manifest": package_manifest_path,
        "activation_manifest": activation_manifest_path,
        "submit": submit_path,
    }


def materialize_release_activation(
    campaign: Campaign,
    *,
    output_root: Path,
    retirement_receipt_path: Path,
    released_utc: str,
) -> dict[str, Path]:
    validate_campaign(campaign, output_root=output_root)
    validate_retirement_receipt(retirement_receipt_path)
    package_dir = output_root / campaign.package_dirname
    frozen_activation_dir = output_root / campaign.activation_dirname
    release_activation_dir = (
        output_root / campaign.release_activation_dirname
    )
    if release_activation_dir.exists():
        raise RepairPackageError(
            f"Refusing to overwrite released {campaign.key} activation."
        )
    release_activation_dir.mkdir(parents=True)
    frozen_manifest_path = (
        frozen_activation_dir / "activation_manifest.json"
    )
    frozen_manifest = load_json(
        frozen_manifest_path, label="frozen v2 activation manifest"
    )
    verify_self_digest(
        frozen_manifest, label="frozen v2 activation manifest"
    )

    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        source = frozen_activation_dir / name
        _exclusive_write(
            release_activation_dir / name,
            source.read_bytes(),
            executable=bool(source.stat().st_mode & 0o111),
        )
    package_relative = package_dir.relative_to(REPO_ROOT)
    release_relative = release_activation_dir.relative_to(REPO_ROOT)
    submit_path = release_activation_dir / "submit.sub"
    _exclusive_write(
        submit_path,
        _submission_text(
            campaign,
            source_sha256=str(
                frozen_manifest["source_archive_sha256"]
            ),
            package_relative=package_relative,
            activation_relative=release_relative,
        ).encode("utf-8"),
    )
    controls = [
        _file_binding(
            release_activation_dir / name,
            relative_to=release_activation_dir,
        )
        for name in (
            "execute_authorized_job.sh",
            "build_attempt_archive.py",
            "submit.sub",
        )
    ]
    control_sha256 = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    retirement_binding = _json_binding(
        retirement_receipt_path.resolve(), relative_to=REPO_ROOT
    )

    authorizations_dir = release_activation_dir / "authorizations"
    authorizations_dir.mkdir()
    release_rows: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    frozen_rows = frozen_manifest.get("executions")
    if not isinstance(frozen_rows, list):
        raise RepairPackageError(
            f"{campaign.key} frozen activation executions drifted."
        )
    for frozen_row in frozen_rows:
        if not isinstance(frozen_row, Mapping):
            raise RepairPackageError("Frozen activation row is malformed.")
        execution_id = str(frozen_row["execution_id"])
        job_path = REPO_ROOT / _safe_relative(
            str(frozen_row["job"]["path"]),
            label="released activation job path",
        )
        frozen_authorization_path = REPO_ROOT / _safe_relative(
            str(frozen_row["authorization"]["path"]),
            label="frozen activation authorization path",
        )
        frozen_authorization = load_json(
            frozen_authorization_path,
            label="frozen v2 authorization",
        )
        verify_self_digest(
            frozen_authorization, label="frozen v2 authorization"
        )
        projection = dict(frozen_authorization)
        projection.pop("sha256", None)
        projection.update(
            {
                "authorization_id": (
                    f"{campaign.release_activation_id}__{execution_id}"
                ),
                "authorized_utc": released_utc,
                "activation_id": campaign.release_activation_id,
                "activation_control_plane_sha256": control_sha256,
                "frozen_predecessor_authorization": _json_binding(
                    frozen_authorization_path,
                    relative_to=REPO_ROOT,
                ),
                "parent_cluster_retirement_receipt": retirement_binding,
                "parent_cluster_retirement_verified": True,
                "parent_cluster_retirement_required_before_thaw": False,
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_not_submitted",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
            }
        )
        authorization = digested(projection)
        authorization_path = (
            authorizations_dir / f"{execution_id}.json"
        )
        _write_json(authorization_path, authorization)
        resources = dict(frozen_row["resources"])
        release_row = {
            "queue_index": int(frozen_row["queue_index"]),
            "parent_proc_id": int(frozen_row["parent_proc_id"]),
            "execution_id": execution_id,
            "job": _json_binding(job_path, relative_to=REPO_ROOT),
            "authorization": _json_binding(
                authorization_path, relative_to=REPO_ROOT
            ),
            "resources": resources,
            "frozen_predecessor_authorization": _json_binding(
                frozen_authorization_path, relative_to=REPO_ROOT
            ),
        }
        release_rows.append(release_row)
        queue_lines.append(
            "\t".join(
                (
                    execution_id,
                    job_path.relative_to(REPO_ROOT).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    sha256_file(authorization_path),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )
    queue_path = release_activation_dir / "queue.tsv"
    _exclusive_write(
        queue_path, "".join(queue_lines).encode("utf-8")
    )

    release_projection = dict(frozen_manifest)
    release_projection.pop("sha256", None)
    release_projection.update(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_release_activation_v2"
            ),
            "activation_id": campaign.release_activation_id,
            "authorized_utc": released_utc,
            "control_plane": controls,
            "activation_control_plane_sha256": control_sha256,
            "executions": release_rows,
            "queue": _file_binding(
                queue_path, relative_to=release_activation_dir
            ),
            "frozen_predecessor_activation": _json_binding(
                frozen_manifest_path, relative_to=REPO_ROOT
            ),
            "parent_cluster_retirement_receipt": retirement_binding,
            "pre_execution_gates": {
                "completed_predecessor_local_preservation_verified": True,
                "parent_cluster_retirement_verified": True,
                "execution_thaw_authorized": True,
                "execution_thaw_blockers": [],
                "submission_starts_frozen": True,
                "post_submit_explicit_thaw_required": True,
            },
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    release_manifest = digested(release_projection)
    release_manifest_path = (
        release_activation_dir / "activation_manifest.json"
    )
    _write_json(release_manifest_path, release_manifest)
    return {
        "release_activation_dir": release_activation_dir,
        "release_activation_manifest": release_manifest_path,
        "submit": submit_path,
    }


def _ordinary_held_scheduler_contract(
    expected_proc_count: int,
) -> dict[str, Any]:
    return {
        "mode": "ordinary_held_exact_proc_release_v1",
        "late_materialization": False,
        "expected_proc_count": expected_proc_count,
        "submit_hold": True,
        "automatic_release": False,
        "post_submit_verification_required_before_release": {
            "exact_cluster_proc_count": expected_proc_count,
            "all_job_status": 5,
            "all_num_job_starts": 0,
        },
        "release_scope": "exact_cluster_proc_only",
        "cluster_wide_release_forbidden": True,
        "owner_wide_release_forbidden": True,
        "constraint_wide_release_forbidden": True,
        "one_proc_per_quota_cycle": True,
    }


def _ordinary_held_pre_execution_gates() -> dict[str, Any]:
    return {
        "completed_predecessor_local_preservation_verified": True,
        "parent_cluster_retirement_verified": True,
        "execution_thaw_authorized": True,
        "execution_thaw_blockers": [],
        "submission_starts_held": True,
        "post_submit_exact_held_readback_required": True,
        "exact_proc_only_release_required": True,
    }


def materialize_ordinary_held_release_activation(
    campaign: Campaign,
    *,
    output_root: Path,
    retirement_receipt_path: Path,
    released_utc: str,
) -> dict[str, Path]:
    validate_release_activation(
        campaign,
        output_root=output_root,
        retirement_receipt_path=retirement_receipt_path,
    )
    package_dir = output_root / campaign.package_dirname
    frozen_activation_dir = output_root / campaign.activation_dirname
    factory_release_dir = (
        output_root / campaign.release_activation_dirname
    )
    held_release_dir = (
        output_root
        / campaign.ordinary_held_release_activation_dirname
    )
    if held_release_dir.exists():
        raise RepairPackageError(
            f"Refusing to overwrite ordinary-held {campaign.key} "
            "release activation."
        )
    held_release_dir.mkdir(parents=True)

    frozen_manifest_path = (
        frozen_activation_dir / "activation_manifest.json"
    )
    factory_release_manifest_path = (
        factory_release_dir / "activation_manifest.json"
    )
    factory_release_manifest = load_json(
        factory_release_manifest_path,
        label="factory release-v1 activation manifest",
    )
    verify_self_digest(
        factory_release_manifest,
        label="factory release-v1 activation manifest",
    )

    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        source = factory_release_dir / name
        _exclusive_write(
            held_release_dir / name,
            source.read_bytes(),
            executable=bool(source.stat().st_mode & 0o111),
        )

    package_relative = package_dir.relative_to(REPO_ROOT)
    held_release_relative = held_release_dir.relative_to(REPO_ROOT)
    submit_path = held_release_dir / "submit.sub"
    _exclusive_write(
        submit_path,
        _ordinary_held_submission_text(
            campaign,
            source_sha256=str(
                factory_release_manifest["source_archive_sha256"]
            ),
            package_relative=package_relative,
            activation_relative=held_release_relative,
        ).encode("utf-8"),
    )
    controls = [
        _file_binding(
            held_release_dir / name,
            relative_to=held_release_dir,
        )
        for name in (
            "execute_authorized_job.sh",
            "build_attempt_archive.py",
            "submit.sub",
        )
    ]
    control_sha256 = hashlib.sha256(
        canonical_json_bytes(controls)
    ).hexdigest()
    retirement_binding = _json_binding(
        retirement_receipt_path.resolve(),
        relative_to=REPO_ROOT,
    )

    authorizations_dir = held_release_dir / "authorizations"
    authorizations_dir.mkdir()
    held_rows: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    predecessor_rows = factory_release_manifest.get("executions")
    if not isinstance(predecessor_rows, list):
        raise RepairPackageError(
            f"{campaign.key} factory release-v1 executions drifted."
        )
    for predecessor_row in predecessor_rows:
        if not isinstance(predecessor_row, Mapping):
            raise RepairPackageError(
                "Factory release-v1 activation row is malformed."
            )
        execution_id = str(predecessor_row["execution_id"])
        job_path = REPO_ROOT / _safe_relative(
            str(predecessor_row["job"]["path"]),
            label="ordinary-held release job path",
        )
        predecessor_authorization_path = (
            REPO_ROOT
            / _safe_relative(
                str(predecessor_row["authorization"]["path"]),
                label="factory release-v1 authorization path",
            )
        )
        predecessor_authorization = load_json(
            predecessor_authorization_path,
            label="factory release-v1 authorization",
        )
        verify_self_digest(
            predecessor_authorization,
            label="factory release-v1 authorization",
        )
        projection = dict(predecessor_authorization)
        projection.pop("sha256", None)
        projection.update(
            {
                "authorization_id": (
                    f"{campaign.ordinary_held_release_activation_id}"
                    f"__{execution_id}"
                ),
                "authorized_utc": released_utc,
                "activation_id": (
                    campaign.ordinary_held_release_activation_id
                ),
                "activation_control_plane_sha256": control_sha256,
                "batch_name": (
                    campaign.ordinary_held_release_batch_name
                ),
                "factory_release_predecessor_authorization": (
                    _json_binding(
                        predecessor_authorization_path,
                        relative_to=REPO_ROOT,
                    )
                ),
                "scheduler_lifecycle_mode": (
                    "ordinary_held_exact_proc_release_v1"
                ),
                "initial_job_status_required": 5,
                "initial_num_job_starts_required": 0,
                "exact_proc_release_required": True,
                "parent_cluster_retirement_receipt": (
                    retirement_binding
                ),
                "parent_cluster_retirement_verified": True,
                "parent_cluster_retirement_required_before_thaw": False,
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_not_submitted",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
            }
        )
        authorization = digested(projection)
        authorization_path = (
            authorizations_dir / f"{execution_id}.json"
        )
        _write_json(authorization_path, authorization)
        resources = dict(predecessor_row["resources"])
        held_row = {
            "queue_index": int(predecessor_row["queue_index"]),
            "parent_proc_id": int(
                predecessor_row["parent_proc_id"]
            ),
            "execution_id": execution_id,
            "job": _json_binding(job_path, relative_to=REPO_ROOT),
            "authorization": _json_binding(
                authorization_path,
                relative_to=REPO_ROOT,
            ),
            "resources": resources,
            "frozen_predecessor_authorization": (
                predecessor_row["frozen_predecessor_authorization"]
            ),
            "factory_release_predecessor_authorization": (
                predecessor_row["authorization"]
            ),
        }
        held_rows.append(held_row)
        queue_lines.append(
            "\t".join(
                (
                    execution_id,
                    job_path.relative_to(REPO_ROOT).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    sha256_file(authorization_path),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )

    queue_path = held_release_dir / "queue.tsv"
    _exclusive_write(
        queue_path,
        "".join(queue_lines).encode("utf-8"),
    )

    held_projection = dict(factory_release_manifest)
    held_projection.pop("sha256", None)
    held_projection.pop("factory_contract", None)
    held_projection.update(
        {
            "schema": (
                "paper_i_checkpoint_retention_subset_"
                "ordinary_held_release_activation_v2"
            ),
            "activation_revision": "release_v2",
            "activation_id": (
                campaign.ordinary_held_release_activation_id
            ),
            "authorized_utc": released_utc,
            "batch_name": campaign.ordinary_held_release_batch_name,
            "control_plane": controls,
            "activation_control_plane_sha256": control_sha256,
            "executions": held_rows,
            "queue": _file_binding(
                queue_path,
                relative_to=held_release_dir,
            ),
            "factory_release_predecessor_activation": (
                _json_binding(
                    factory_release_manifest_path,
                    relative_to=REPO_ROOT,
                )
            ),
            "frozen_predecessor_activation": _json_binding(
                frozen_manifest_path,
                relative_to=REPO_ROOT,
            ),
            "parent_cluster_retirement_receipt": retirement_binding,
            "scheduler_contract": (
                _ordinary_held_scheduler_contract(len(held_rows))
            ),
            "scheduler_revision": {
                "from_mode": "bounded_factory_frozen_max_idle_zero",
                "to_mode": "ordinary_held_exact_proc_release_v1",
                "reason": (
                    "chtc_schedd_factory_submission_path_failed_"
                    "while_ordinary_held_smoke_succeeded"
                ),
                "scientific_job_or_runtime_change": False,
            },
            "pre_execution_gates": (
                _ordinary_held_pre_execution_gates()
            ),
            "submission_initial_state": (
                "ordinary_all_procs_held_before_start"
            ),
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "authorized_not_submitted",
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    held_manifest = digested(held_projection)
    held_manifest_path = (
        held_release_dir / "activation_manifest.json"
    )
    _write_json(held_manifest_path, held_manifest)
    return {
        "release_activation_dir": held_release_dir,
        "release_activation_manifest": held_manifest_path,
        "submit": submit_path,
    }


_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<name>[+A-Za-z_][+.A-Za-z0-9_]*)\s*=\s*(?P<value>.*?)\s*$"
)


def _submit_assignments(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ASSIGNMENT_RE.match(line)
        if match:
            result.setdefault(match.group("name").lower(), []).append(
                match.group("value").strip()
            )
    return result


def validate_campaign(
    campaign: Campaign,
    *,
    output_root: Path,
) -> dict[str, Any]:
    context = _parent_context(campaign)
    preservation_snapshot_path = (
        REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE
    )
    preservation_snapshot = validate_preservation_snapshot(
        preservation_snapshot_path
    )
    preservation_snapshot_binding = _json_binding(
        preservation_snapshot_path, relative_to=REPO_ROOT
    )
    package_dir = output_root / campaign.package_dirname
    activation_dir = output_root / campaign.activation_dirname
    package_manifest = load_json(
        package_dir / "package_manifest.json",
        label="v2 package manifest",
    )
    activation_manifest = load_json(
        activation_dir / "activation_manifest.json",
        label="v2 activation manifest",
    )
    for label, payload in (
        ("v2 package manifest", package_manifest),
        ("v2 activation manifest", activation_manifest),
    ):
        verify_self_digest(payload, label=label)
    if (
        package_manifest.get("operational_package_id")
        != campaign.operational_package_id
        or activation_manifest.get("activation_id")
        != campaign.activation_id
        or package_manifest.get(
            "completed_v1_preservation_snapshot"
        )
        != preservation_snapshot_binding
        or activation_manifest.get(
            "completed_v1_preservation_snapshot"
        )
        != preservation_snapshot_binding
    ):
        raise RepairPackageError(
            f"{campaign.key} v2 identity drifted."
        )

    parent_order, parent_payloads, parent_metadata = _read_parent_source(
        campaign, context
    )
    repaired_manifest = load_json(
        package_dir / "source_archive_manifest.json",
        label="v2 source archive manifest",
    )
    delta = load_json(
        package_dir / "source_delta_receipt.json",
        label="v2 source delta receipt",
    )
    supersession = load_json(
        package_dir / "supersession_map.json",
        label="v2 supersession map",
    )
    plan = load_json(
        package_dir / "execution_plan.json",
        label="v2 execution plan",
    )
    for label, payload in (
        ("v2 source archive manifest", repaired_manifest),
        ("v2 source delta receipt", delta),
        ("v2 supersession map", supersession),
        ("v2 execution plan", plan),
    ):
        verify_self_digest(payload, label=label)
    repaired_archive = package_dir / "source_locked.tar.gz"
    if (
        repaired_manifest.get("archive", {}).get("sha256")
        != sha256_file(repaired_archive)
        or delta.get("changed_member_count") != 1
        or delta.get("changed_members", [{}])[0].get("path")
        != CHECKPOINT_MEMBER
        or delta.get("changed_members", [{}])[0].get("parent_sha256")
        != PARENT_CHECKPOINT_SHA256
        or delta.get("changed_members", [{}])[0].get("repaired_sha256")
        != REPAIRED_CHECKPOINT_SHA256
    ):
        raise RepairPackageError(
            f"{campaign.key} v2 source delta receipt drifted."
        )
    repaired_payloads: dict[str, bytes] = {}
    repaired_modes: dict[str, int] = {}
    repaired_order: list[str] = []
    with tarfile.open(repaired_archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.issym() or member.islnk():
                raise RepairPackageError(
                    "v2 source archive has a non-regular member."
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise RepairPackageError(
                    f"Unreadable v2 source member: {member.name}"
                )
            repaired_order.append(member.name)
            repaired_payloads[member.name] = stream.read()
            repaired_modes[member.name] = int(member.mode)
    changed = [
        name
        for name in parent_order
        if parent_payloads[name] != repaired_payloads.get(name)
    ]
    if (
        repaired_order != parent_order
        or changed != [CHECKPOINT_MEMBER]
        or repaired_modes
        != {
            name: int(parent_metadata[name]["mode"])
            for name in parent_order
        }
        or sha256_bytes(repaired_payloads[CHECKPOINT_MEMBER])
        != REPAIRED_CHECKPOINT_SHA256
        or (
            campaign.expected_source_archive_sha256 is not None
            and sha256_file(repaired_archive)
            != campaign.expected_source_archive_sha256
        )
    ):
        raise RepairPackageError(
            f"{campaign.key} source archive is not a one-member delta."
        )

    parent_execution_ids = [
        str(row["execution_id"]) for row in context["executions"]
    ]
    expected_remaining = [
        execution_id
        for proc_id, execution_id in enumerate(parent_execution_ids)
        if proc_id not in set(campaign.completed_proc_ids)
    ]
    if (
        plan.get("execution_ids") != expected_remaining
        or plan.get("completed_v1_preservation_snapshot")
        != preservation_snapshot_binding
        or plan.get("campaign_cell_count")
        != campaign.campaign_cell_count
        or int(package_manifest.get("direct_execution_count", -1))
        != len(expected_remaining)
        or package_manifest.get("campaign_cell_count")
        != campaign.campaign_cell_count
        or supersession.get("completed_parent_proc_ids")
        != list(campaign.completed_proc_ids)
        or supersession.get("completed_v1_preservation_snapshot")
        != preservation_snapshot_binding
        or len(supersession.get("rows", []))
        != len(parent_execution_ids)
    ):
        raise RepairPackageError(
            f"{campaign.key} v2 subset/supersession drifted."
        )
    snapshot_rows = [
        row
        for row in preservation_snapshot["rows"]
        if row["campaign_key"] == campaign.key
    ]
    completed_supersession_rows = [
        row
        for row in supersession["rows"]
        if row["state"] == "completed_v1_preserved_excluded_from_v2"
    ]
    if len(completed_supersession_rows) != len(snapshot_rows):
        raise RepairPackageError(
            f"{campaign.key} completed snapshot closure drifted."
        )
    for supersession_row, snapshot_row in zip(
        completed_supersession_rows, snapshot_rows, strict=True
    ):
        if supersession_row.get("completion_evidence") != {
            "preservation_snapshot": preservation_snapshot_binding,
            "snapshot_row": snapshot_row,
        }:
            raise RepairPackageError(
                f"{campaign.key} immutable completion evidence drifted."
            )
    if "rolling_receipt" in canonical_json_bytes(supersession).decode(
        "ascii"
    ):
        raise RepairPackageError(
            f"{campaign.key} supersession binds mutable rolling state."
        )
    jobs = package_manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(expected_remaining):
        raise RepairPackageError(
            f"{campaign.key} v2 job set drifted."
        )
    for row in jobs:
        if not isinstance(row, Mapping):
            raise RepairPackageError("Malformed v2 job row.")
        copied = package_dir / str(row["job"]["path"])
        parent = REPO_ROOT / str(row["parent_job"]["path"])
        if copied.read_bytes() != parent.read_bytes():
            raise RepairPackageError(
                f"Scientific parent job bytes drifted: {copied.name}"
            )
    for name in ("run_cell.py", "package_contract.py"):
        if (
            (package_dir / name).read_bytes()
            != (campaign.parent_package / name).read_bytes()
        ):
            raise RepairPackageError(
                f"Parent runtime control drifted: {name}"
            )
    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        if (
            (activation_dir / name).read_bytes()
            != (campaign.parent_activation / name).read_bytes()
        ):
            raise RepairPackageError(
                f"Parent activation runtime drifted: {name}"
            )

    submit_text = (activation_dir / "submit.sub").read_text(
        encoding="utf-8"
    )
    assignments = _submit_assignments(submit_text)
    if (
        assignments.get("max_materialize")
        != [str(campaign.max_materialize)]
        or assignments.get("max_idle") != ["0"]
        or assignments.get("leave_in_queue")
        != ["(JobStatus == 4) && (ExitCode =!= 0)"]
    ):
        raise RepairPackageError(
            f"{campaign.key} factory lifecycle is not frozen/safe."
        )
    if (
        activation_manifest.get("direct_execution_count")
        != len(expected_remaining)
        or activation_manifest.get("campaign_cell_count")
        != campaign.campaign_cell_count
        or activation_manifest.get("factory_contract", {}).get(
            "successful_rows_reap"
        )
        is not True
        or activation_manifest.get("factory_contract", {}).get(
            "max_idle"
        )
        != 0
        or activation_manifest.get("pre_execution_gates", {}).get(
            "completed_predecessor_local_preservation_verified"
        )
        is not True
        or activation_manifest.get("pre_execution_gates", {}).get(
            "parent_cluster_retirement_verified"
        )
        is not False
        or activation_manifest.get("pre_execution_gates", {}).get(
            "execution_thaw_authorized"
        )
        is not False
    ):
        raise RepairPackageError(
            f"{campaign.key} activation lifecycle drifted."
        )
    activation_rows = activation_manifest.get("executions")
    if (
        not isinstance(activation_rows, list)
        or [row["execution_id"] for row in activation_rows]
        != expected_remaining
    ):
        raise RepairPackageError(
            f"{campaign.key} activation execution order drifted."
        )
    for row in activation_rows:
        authorization_path = REPO_ROOT / str(
            row["authorization"]["path"]
        )
        authorization = load_json(
            authorization_path, label="v2 authorization"
        )
        verify_self_digest(authorization, label="v2 authorization")
        if (
            authorization.get("activation_id")
            != campaign.activation_id
            or authorization.get("source_archive_sha256")
            != repaired_manifest["archive"]["sha256"]
            or authorization.get("scientific_settings_changed") != []
            or authorization.get(
                "completed_v1_preservation_snapshot"
            )
            != preservation_snapshot_binding
            or authorization.get("execution_authorized") is not True
            or authorization.get("submission_authorized") is not True
        ):
            raise RepairPackageError(
                f"Authorization drifted: {row['execution_id']}"
            )
    expected_package_files = {
        package_dir / "source_locked.tar.gz",
        package_dir / "source_archive_manifest.json",
        package_dir / "source_delta_receipt.json",
        package_dir / "execution_plan.json",
        package_dir / "supersession_map.json",
        package_dir / "queue.tsv",
        package_dir / "run_cell.py",
        package_dir / "package_contract.py",
        package_dir / "package_manifest.json",
        *(
            package_dir / str(row["job"]["path"])
            for row in jobs
        ),
    }
    observed_package_files = {
        path for path in package_dir.rglob("*") if path.is_file()
    }
    expected_activation_files = {
        activation_dir / "execute_authorized_job.sh",
        activation_dir / "build_attempt_archive.py",
        activation_dir / "submit.sub",
        activation_dir / "queue.tsv",
        activation_dir / "activation_manifest.json",
        *(
            REPO_ROOT / str(row["authorization"]["path"])
            for row in activation_rows
        ),
    }
    observed_activation_files = {
        path for path in activation_dir.rglob("*") if path.is_file()
    }
    if (
        observed_package_files != expected_package_files
        or observed_activation_files != expected_activation_files
        or any(path.is_symlink() for path in package_dir.rglob("*"))
        or any(path.is_symlink() for path in activation_dir.rglob("*"))
    ):
        raise RepairPackageError(
            f"{campaign.key} recursive package/activation closure drifted."
        )
    return {
        "campaign": campaign.key,
        "status": "passed",
        "operational_package_id": campaign.operational_package_id,
        "activation_id": campaign.activation_id,
        "parent_row_count": len(parent_execution_ids),
        "completed_v1_excluded_count": len(
            campaign.completed_proc_ids
        ),
        "v2_queue_count": len(expected_remaining),
        "source_archive_sha256": repaired_manifest["archive"]["sha256"],
        "changed_source_members": changed,
        "scientific_settings_changed": [],
        "factory_initial_state": "frozen_no_rows_materialized",
    }


def validate_release_activation(
    campaign: Campaign,
    *,
    output_root: Path,
    retirement_receipt_path: Path,
) -> dict[str, Any]:
    validate_campaign(campaign, output_root=output_root)
    retirement = validate_retirement_receipt(
        retirement_receipt_path
    )
    preservation_snapshot_path = (
        REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE
    )
    validate_preservation_snapshot(preservation_snapshot_path)
    preservation_snapshot_binding = _json_binding(
        preservation_snapshot_path, relative_to=REPO_ROOT
    )
    package_dir = output_root / campaign.package_dirname
    frozen_activation_dir = output_root / campaign.activation_dirname
    release_activation_dir = (
        output_root / campaign.release_activation_dirname
    )
    frozen_manifest_path = (
        frozen_activation_dir / "activation_manifest.json"
    )
    frozen_manifest = load_json(
        frozen_manifest_path, label="frozen v2 activation manifest"
    )
    release_manifest_path = (
        release_activation_dir / "activation_manifest.json"
    )
    release_manifest = load_json(
        release_manifest_path, label="released v2 activation manifest"
    )
    verify_self_digest(
        release_manifest, label="released v2 activation manifest"
    )
    expected_retirement_binding = _json_binding(
        retirement_receipt_path.resolve(), relative_to=REPO_ROOT
    )
    expected_frozen_binding = _json_binding(
        frozen_manifest_path, relative_to=REPO_ROOT
    )
    expected_ids = [
        row["execution_id"] for row in frozen_manifest["executions"]
    ]
    if (
        release_manifest.get("schema")
        != "paper_i_checkpoint_retention_subset_release_activation_v2"
        or release_manifest.get("activation_id")
        != campaign.release_activation_id
        or release_manifest.get("operational_package_id")
        != campaign.operational_package_id
        or release_manifest.get("campaign_id")
        != frozen_manifest.get("campaign_id")
        or release_manifest.get("source_archive_sha256")
        != frozen_manifest.get("source_archive_sha256")
        or release_manifest.get("source_delta_receipt")
        != frozen_manifest.get("source_delta_receipt")
        or release_manifest.get("supersession_map")
        != frozen_manifest.get("supersession_map")
        or release_manifest.get(
            "completed_v1_preservation_snapshot"
        )
        != preservation_snapshot_binding
        or frozen_manifest.get(
            "completed_v1_preservation_snapshot"
        )
        != preservation_snapshot_binding
        or release_manifest.get("sealed_operational_package")
        != frozen_manifest.get("sealed_operational_package")
        or release_manifest.get("frozen_predecessor_activation")
        != expected_frozen_binding
        or release_manifest.get("parent_cluster_retirement_receipt")
        != expected_retirement_binding
        or release_manifest.get("pre_execution_gates")
        != {
            "completed_predecessor_local_preservation_verified": True,
            "parent_cluster_retirement_verified": True,
            "execution_thaw_authorized": True,
            "execution_thaw_blockers": [],
            "submission_starts_frozen": True,
            "post_submit_explicit_thaw_required": True,
        }
        or release_manifest.get("execution_authorized") is not True
        or release_manifest.get("submission_authorized") is not True
        or release_manifest.get("submitted") is not False
    ):
        raise RepairPackageError(
            f"{campaign.key} release transition drifted."
        )
    controls = release_manifest.get("control_plane")
    if not isinstance(controls, list):
        raise RepairPackageError("Release control plane is absent.")
    for binding in controls:
        if not isinstance(binding, Mapping):
            raise RepairPackageError(
                "Release control binding is malformed."
            )
        path = release_activation_dir / _safe_relative(
            str(binding.get("path", "")),
            label="release control path",
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != binding.get("sha256")
            or path.stat().st_size != int(binding.get("size_bytes", -1))
        ):
            raise RepairPackageError(
                f"Release control drifted: {path.name}"
            )
    if hashlib.sha256(canonical_json_bytes(controls)).hexdigest() != (
        release_manifest.get("activation_control_plane_sha256")
    ):
        raise RepairPackageError("Release control-plane digest drifted.")
    submit_assignments = _submit_assignments(
        (release_activation_dir / "submit.sub").read_text(
            encoding="utf-8"
        )
    )
    if (
        submit_assignments.get("max_materialize")
        != [str(campaign.max_materialize)]
        or submit_assignments.get("max_idle") != ["0"]
        or submit_assignments.get("leave_in_queue")
        != ["(JobStatus == 4) && (ExitCode =!= 0)"]
    ):
        raise RepairPackageError(
            f"{campaign.key} released factory lifecycle drifted."
        )
    release_rows = release_manifest.get("executions")
    if (
        not isinstance(release_rows, list)
        or [row["execution_id"] for row in release_rows]
        != expected_ids
    ):
        raise RepairPackageError(
            f"{campaign.key} released execution order drifted."
        )
    expected_files = {
        release_activation_dir / "execute_authorized_job.sh",
        release_activation_dir / "build_attempt_archive.py",
        release_activation_dir / "submit.sub",
        release_activation_dir / "queue.tsv",
        release_manifest_path,
    }
    queue_lines: list[str] = []
    for index, (release_row, frozen_row) in enumerate(
        zip(release_rows, frozen_manifest["executions"], strict=True)
    ):
        if (
            not isinstance(release_row, Mapping)
            or not isinstance(frozen_row, Mapping)
            or int(release_row.get("queue_index", -1)) != index
            or release_row.get("execution_id")
            != frozen_row.get("execution_id")
            or release_row.get("job") != frozen_row.get("job")
            or release_row.get("resources") != frozen_row.get("resources")
            or release_row.get("frozen_predecessor_authorization")
            != frozen_row.get("authorization")
        ):
            raise RepairPackageError(
                f"{campaign.key} released row drifted at {index}."
            )
        authorization_path = REPO_ROOT / _safe_relative(
            str(release_row["authorization"]["path"]),
            label="release authorization path",
        )
        expected_files.add(authorization_path)
        authorization = load_json(
            authorization_path, label="released v2 authorization"
        )
        verify_self_digest(
            authorization, label="released v2 authorization"
        )
        job_path = REPO_ROOT / _safe_relative(
            str(release_row["job"]["path"]),
            label="release job path",
        )
        job = load_json(job_path, label="released job")
        resources = release_row["resources"]
        if (
            authorization.get("activation_id")
            != campaign.release_activation_id
            or authorization.get("execution_id")
            != release_row["execution_id"]
            or authorization.get("job_sha256") != job["sha256"]
            or authorization.get("job_file_sha256")
            != sha256_file(job_path)
            or authorization.get("source_archive_sha256")
            != release_manifest["source_archive_sha256"]
            or authorization.get("parent_cluster_retirement_receipt")
            != expected_retirement_binding
            or authorization.get(
                "completed_v1_preservation_snapshot"
            )
            != preservation_snapshot_binding
            or authorization.get(
                "parent_cluster_retirement_verified"
            )
            is not True
            or authorization.get(
                "parent_cluster_retirement_required_before_thaw"
            )
            is not False
            or authorization.get("execution_authorized") is not True
            or authorization.get("submission_authorized") is not True
        ):
            raise RepairPackageError(
                f"Released authorization drifted: "
                f"{release_row['execution_id']}"
            )
        queue_lines.append(
            "\t".join(
                (
                    str(release_row["execution_id"]),
                    job_path.relative_to(REPO_ROOT).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    sha256_file(authorization_path),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )
    queue_path = release_activation_dir / "queue.tsv"
    if queue_path.read_text(encoding="utf-8") != "".join(queue_lines):
        raise RepairPackageError(
            f"{campaign.key} released queue drifted."
        )
    observed_files = {
        path
        for path in release_activation_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise RepairPackageError(
            f"{campaign.key} release recursive closure drifted."
        )
    if not package_dir.is_dir():
        raise RepairPackageError("Released activation lost its package.")
    return {
        "campaign": campaign.key,
        "status": "passed",
        "release_activation_id": campaign.release_activation_id,
        "direct_execution_count": len(expected_ids),
        "parent_cluster_ids": retirement["target_cluster_ids"],
        "parent_cluster_retirement_verified": True,
        "execution_thaw_authorized": True,
        "submission_initial_state": "frozen_max_idle_zero",
    }


def validate_ordinary_held_release_activation(
    campaign: Campaign,
    *,
    output_root: Path,
    retirement_receipt_path: Path,
) -> dict[str, Any]:
    validate_release_activation(
        campaign,
        output_root=output_root,
        retirement_receipt_path=retirement_receipt_path,
    )
    retirement = validate_retirement_receipt(
        retirement_receipt_path
    )
    preservation_snapshot_path = (
        REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE
    )
    validate_preservation_snapshot(preservation_snapshot_path)
    preservation_snapshot_binding = _json_binding(
        preservation_snapshot_path,
        relative_to=REPO_ROOT,
    )
    package_dir = output_root / campaign.package_dirname
    frozen_activation_dir = output_root / campaign.activation_dirname
    factory_release_dir = (
        output_root / campaign.release_activation_dirname
    )
    held_release_dir = (
        output_root
        / campaign.ordinary_held_release_activation_dirname
    )
    frozen_manifest_path = (
        frozen_activation_dir / "activation_manifest.json"
    )
    factory_release_manifest_path = (
        factory_release_dir / "activation_manifest.json"
    )
    held_manifest_path = (
        held_release_dir / "activation_manifest.json"
    )
    factory_release_manifest = load_json(
        factory_release_manifest_path,
        label="factory release-v1 activation manifest",
    )
    held_manifest = load_json(
        held_manifest_path,
        label="ordinary-held release-v2 activation manifest",
    )
    verify_self_digest(
        held_manifest,
        label="ordinary-held release-v2 activation manifest",
    )
    expected_retirement_binding = _json_binding(
        retirement_receipt_path.resolve(),
        relative_to=REPO_ROOT,
    )
    expected_frozen_binding = _json_binding(
        frozen_manifest_path,
        relative_to=REPO_ROOT,
    )
    expected_factory_release_binding = _json_binding(
        factory_release_manifest_path,
        relative_to=REPO_ROOT,
    )
    predecessor_rows = factory_release_manifest.get("executions")
    if not isinstance(predecessor_rows, list):
        raise RepairPackageError(
            f"{campaign.key} factory release-v1 executions drifted."
        )
    expected_ids = [
        row["execution_id"] for row in predecessor_rows
    ]
    expected_scheduler_contract = _ordinary_held_scheduler_contract(
        len(expected_ids)
    )
    unchanged_fields = (
        "operational_package_id",
        "campaign_id",
        "source_archive_sha256",
        "source_delta_receipt",
        "supersession_map",
        "completed_v1_preservation_snapshot",
        "sealed_operational_package",
        "parent_source_archive_sha256",
        "scientific_job_bytes_identical",
        "scientific_protocol_bytes_identical",
        "scientific_settings_changed",
        "direct_execution_count",
        "campaign_cell_count",
        "queue_variables",
        "remote_image",
    )
    if any(
        held_manifest.get(field)
        != factory_release_manifest.get(field)
        for field in unchanged_fields
    ):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held scientific projection "
            "drifted."
        )
    if (
        held_manifest.get("schema")
        != (
            "paper_i_checkpoint_retention_subset_"
            "ordinary_held_release_activation_v2"
        )
        or held_manifest.get("activation_revision") != "release_v2"
        or held_manifest.get("activation_id")
        != campaign.ordinary_held_release_activation_id
        or held_manifest.get("batch_name")
        != campaign.ordinary_held_release_batch_name
        or held_manifest.get("frozen_predecessor_activation")
        != expected_frozen_binding
        or held_manifest.get(
            "factory_release_predecessor_activation"
        )
        != expected_factory_release_binding
        or held_manifest.get("parent_cluster_retirement_receipt")
        != expected_retirement_binding
        or held_manifest.get("completed_v1_preservation_snapshot")
        != preservation_snapshot_binding
        or held_manifest.get("scheduler_contract")
        != expected_scheduler_contract
        or held_manifest.get("scheduler_revision")
        != {
            "from_mode": "bounded_factory_frozen_max_idle_zero",
            "to_mode": "ordinary_held_exact_proc_release_v1",
            "reason": (
                "chtc_schedd_factory_submission_path_failed_"
                "while_ordinary_held_smoke_succeeded"
            ),
            "scientific_job_or_runtime_change": False,
        }
        or held_manifest.get("pre_execution_gates")
        != _ordinary_held_pre_execution_gates()
        or held_manifest.get("submission_initial_state")
        != "ordinary_all_procs_held_before_start"
        or held_manifest.get("execution_authorized") is not True
        or held_manifest.get("submission_authorized") is not True
        or held_manifest.get("submission_state")
        != "authorized_not_submitted"
        or held_manifest.get("submitted") is not False
        or "factory_contract" in held_manifest
    ):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held release transition "
            "drifted."
        )

    controls = held_manifest.get("control_plane")
    if not isinstance(controls, list):
        raise RepairPackageError(
            "Ordinary-held release control plane is absent."
        )
    for binding in controls:
        if not isinstance(binding, Mapping):
            raise RepairPackageError(
                "Ordinary-held release control binding is malformed."
            )
        path = held_release_dir / _safe_relative(
            str(binding.get("path", "")),
            label="ordinary-held release control path",
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != binding.get("sha256")
            or path.stat().st_size
            != int(binding.get("size_bytes", -1))
        ):
            raise RepairPackageError(
                f"Ordinary-held release control drifted: "
                f"{path.name}"
            )
    if hashlib.sha256(canonical_json_bytes(controls)).hexdigest() != (
        held_manifest.get("activation_control_plane_sha256")
    ):
        raise RepairPackageError(
            "Ordinary-held release control-plane digest drifted."
        )
    for name in (
        "execute_authorized_job.sh",
        "build_attempt_archive.py",
    ):
        if (
            (held_release_dir / name).read_bytes()
            != (factory_release_dir / name).read_bytes()
            or (held_release_dir / name).read_bytes()
            != (frozen_activation_dir / name).read_bytes()
        ):
            raise RepairPackageError(
                f"Ordinary-held runtime control drifted: {name}"
            )

    submit_assignments = _submit_assignments(
        (held_release_dir / "submit.sub").read_text(
            encoding="utf-8"
        )
    )
    if (
        "max_materialize" in submit_assignments
        or "max_idle" in submit_assignments
        or submit_assignments.get("hold") != ["True"]
        or submit_assignments.get("periodic_release") != ["False"]
        or submit_assignments.get("+holsteinlifecyclemode")
        != ['"ordinary_held_exact_proc_release_v1"']
        or submit_assignments.get("+jobbatchname")
        != [f'"{campaign.ordinary_held_release_batch_name}"']
        or submit_assignments.get("leave_in_queue")
        != ["(JobStatus == 4) && (ExitCode =!= 0)"]
    ):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held submit lifecycle drifted."
        )

    held_rows = held_manifest.get("executions")
    if (
        not isinstance(held_rows, list)
        or [row["execution_id"] for row in held_rows]
        != expected_ids
        or len(held_rows)
        != (
            campaign.campaign_cell_count
            - len(campaign.completed_proc_ids)
        )
    ):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held execution order drifted."
        )
    expected_files = {
        held_release_dir / "execute_authorized_job.sh",
        held_release_dir / "build_attempt_archive.py",
        held_release_dir / "submit.sub",
        held_release_dir / "queue.tsv",
        held_manifest_path,
    }
    queue_lines: list[str] = []
    for index, (held_row, predecessor_row) in enumerate(
        zip(held_rows, predecessor_rows, strict=True)
    ):
        if (
            not isinstance(held_row, Mapping)
            or not isinstance(predecessor_row, Mapping)
            or int(held_row.get("queue_index", -1)) != index
            or held_row.get("execution_id")
            != predecessor_row.get("execution_id")
            or held_row.get("parent_proc_id")
            != predecessor_row.get("parent_proc_id")
            or held_row.get("job") != predecessor_row.get("job")
            or held_row.get("resources")
            != predecessor_row.get("resources")
            or held_row.get("frozen_predecessor_authorization")
            != predecessor_row.get(
                "frozen_predecessor_authorization"
            )
            or held_row.get(
                "factory_release_predecessor_authorization"
            )
            != predecessor_row.get("authorization")
        ):
            raise RepairPackageError(
                f"{campaign.key} ordinary-held row drifted at "
                f"{index}."
            )
        authorization_path = REPO_ROOT / _safe_relative(
            str(held_row["authorization"]["path"]),
            label="ordinary-held release authorization path",
        )
        predecessor_authorization_path = (
            REPO_ROOT
            / _safe_relative(
                str(predecessor_row["authorization"]["path"]),
                label="factory release-v1 authorization path",
            )
        )
        expected_predecessor_authorization = _json_binding(
            predecessor_authorization_path,
            relative_to=REPO_ROOT,
        )
        expected_files.add(authorization_path)
        authorization = load_json(
            authorization_path,
            label="ordinary-held release-v2 authorization",
        )
        verify_self_digest(
            authorization,
            label="ordinary-held release-v2 authorization",
        )
        job_path = REPO_ROOT / _safe_relative(
            str(held_row["job"]["path"]),
            label="ordinary-held release job path",
        )
        job = load_json(job_path, label="ordinary-held released job")
        resources = held_row["resources"]
        if (
            authorization.get("activation_id")
            != campaign.ordinary_held_release_activation_id
            or authorization.get("authorization_id")
            != (
                f"{campaign.ordinary_held_release_activation_id}"
                f"__{held_row['execution_id']}"
            )
            or authorization.get("execution_id")
            != held_row["execution_id"]
            or authorization.get("job_sha256") != job["sha256"]
            or authorization.get("job_file_sha256")
            != sha256_file(job_path)
            or authorization.get("source_archive_sha256")
            != held_manifest["source_archive_sha256"]
            or authorization.get(
                "factory_release_predecessor_authorization"
            )
            != expected_predecessor_authorization
            or authorization.get("frozen_predecessor_authorization")
            != held_row["frozen_predecessor_authorization"]
            or authorization.get("parent_cluster_retirement_receipt")
            != expected_retirement_binding
            or authorization.get(
                "completed_v1_preservation_snapshot"
            )
            != preservation_snapshot_binding
            or authorization.get(
                "parent_cluster_retirement_verified"
            )
            is not True
            or authorization.get(
                "parent_cluster_retirement_required_before_thaw"
            )
            is not False
            or authorization.get("scheduler_lifecycle_mode")
            != "ordinary_held_exact_proc_release_v1"
            or authorization.get("initial_job_status_required") != 5
            or authorization.get(
                "initial_num_job_starts_required"
            )
            != 0
            or authorization.get("exact_proc_release_required")
            is not True
            or authorization.get("batch_name")
            != campaign.ordinary_held_release_batch_name
            or authorization.get("execution_authorized") is not True
            or authorization.get("submission_authorized") is not True
        ):
            raise RepairPackageError(
                f"Ordinary-held authorization drifted: "
                f"{held_row['execution_id']}"
            )
        queue_lines.append(
            "\t".join(
                (
                    str(held_row["execution_id"]),
                    job_path.relative_to(REPO_ROOT).as_posix(),
                    sha256_file(job_path),
                    authorization_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    sha256_file(authorization_path),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
            + "\n"
        )
    queue_path = held_release_dir / "queue.tsv"
    if queue_path.read_text(encoding="utf-8") != "".join(queue_lines):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held queue drifted."
        )
    observed_files = {
        path for path in held_release_dir.rglob("*") if path.is_file()
    }
    if (
        observed_files != expected_files
        or any(path.is_symlink() for path in held_release_dir.rglob("*"))
    ):
        raise RepairPackageError(
            f"{campaign.key} ordinary-held recursive closure "
            "drifted."
        )
    if not package_dir.is_dir():
        raise RepairPackageError(
            "Ordinary-held activation lost its package."
        )
    return {
        "campaign": campaign.key,
        "status": "passed",
        "release_activation_id": (
            campaign.ordinary_held_release_activation_id
        ),
        "direct_execution_count": len(expected_ids),
        "parent_cluster_ids": retirement["target_cluster_ids"],
        "parent_cluster_retirement_verified": True,
        "execution_thaw_authorized": True,
        "submission_initial_state": (
            "ordinary_all_procs_held_before_start"
        ),
    }


def _selected_campaigns(key: str) -> tuple[Campaign, ...]:
    if key == "all":
        return CAMPAIGNS
    selected = tuple(campaign for campaign in CAMPAIGNS if campaign.key == key)
    if not selected:
        raise RepairPackageError(f"Unknown campaign: {key}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument(
        "--authorized-utc",
        required=True,
        help="Explicit UTC timestamp recorded in the new authorizations.",
    )
    materialize.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    materialize.add_argument(
        "--output-root", type=Path, default=BASE
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    validate.add_argument(
        "--output-root", type=Path, default=BASE
    )
    preservation = subparsers.add_parser(
        "seal-preservation-snapshot"
    )
    preservation.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE,
    )
    preservation.add_argument("--sealed-utc", required=True)
    preservation_validate = subparsers.add_parser(
        "validate-preservation-snapshot"
    )
    preservation_validate.add_argument(
        "--snapshot",
        type=Path,
        default=REPO_ROOT / PRESERVATION_SNAPSHOT_RELATIVE,
    )
    retirement = subparsers.add_parser("seal-retirement-receipt")
    retirement.add_argument("--output", type=Path, required=True)
    retirement.add_argument("--retired-utc", required=True)
    retirement.add_argument("--verified-utc", required=True)
    retirement.add_argument("--schedd", required=True)
    retirement.add_argument("--owner", required=True)
    retirement.add_argument(
        "--condor-rm-stdout", type=Path, required=True
    )
    retirement.add_argument(
        "--post-condor-q-json", type=Path, required=True
    )
    retirement.add_argument(
        "--post-condor-q-factory-json", type=Path, required=True
    )
    retirement.add_argument(
        "--condor-rm-exit-status", type=int, required=True
    )
    retirement.add_argument(
        "--post-condor-q-exit-status", type=int, required=True
    )
    retirement.add_argument(
        "--post-condor-q-factory-exit-status",
        type=int,
        required=True,
    )
    release = subparsers.add_parser("release")
    release.add_argument("--retirement-receipt", type=Path, required=True)
    release.add_argument("--released-utc", required=True)
    release.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    release.add_argument("--output-root", type=Path, default=BASE)
    validate_release = subparsers.add_parser("validate-release")
    validate_release.add_argument(
        "--retirement-receipt", type=Path, required=True
    )
    validate_release.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    validate_release.add_argument(
        "--output-root", type=Path, default=BASE
    )
    held_release = subparsers.add_parser("release-held-v2")
    held_release.add_argument(
        "--retirement-receipt", type=Path, required=True
    )
    held_release.add_argument("--released-utc", required=True)
    held_release.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    held_release.add_argument(
        "--output-root", type=Path, default=BASE
    )
    validate_held_release = subparsers.add_parser(
        "validate-release-held-v2"
    )
    validate_held_release.add_argument(
        "--retirement-receipt", type=Path, required=True
    )
    validate_held_release.add_argument(
        "--campaign",
        choices=("all", *(campaign.key for campaign in CAMPAIGNS)),
        default="all",
    )
    validate_held_release.add_argument(
        "--output-root", type=Path, default=BASE
    )
    args = parser.parse_args()
    if args.action == "seal-preservation-snapshot":
        snapshot = seal_preservation_snapshot(
            output_path=args.output,
            sealed_utc=args.sealed_utc,
        )
        print(
            canonical_json_bytes(
                {
                    "status": "passed",
                    "preservation_snapshot": (
                        args.output.resolve().as_posix()
                    ),
                    "sha256": snapshot["sha256"],
                }
            ).decode("ascii")
        )
        return 0
    if args.action == "validate-preservation-snapshot":
        snapshot = validate_preservation_snapshot(
            args.snapshot.resolve()
        )
        print(
            canonical_json_bytes(
                {
                    "status": "passed",
                    "preservation_snapshot": (
                        args.snapshot.resolve().as_posix()
                    ),
                    "sha256": snapshot["sha256"],
                }
            ).decode("ascii")
        )
        return 0
    if args.action == "seal-retirement-receipt":
        receipt = seal_retirement_receipt(
            output_path=args.output,
            retired_utc=args.retired_utc,
            verified_utc=args.verified_utc,
            schedd=args.schedd,
            owner=args.owner,
            condor_rm_stdout=args.condor_rm_stdout,
            post_condor_q_json=args.post_condor_q_json,
            post_condor_q_factory_json=(
                args.post_condor_q_factory_json
            ),
            condor_rm_exit_status=args.condor_rm_exit_status,
            post_condor_q_exit_status=(
                args.post_condor_q_exit_status
            ),
            post_condor_q_factory_exit_status=(
                args.post_condor_q_factory_exit_status
            ),
        )
        print(
            canonical_json_bytes(
                {
                    "status": "passed",
                    "retirement_receipt": args.output.resolve().as_posix(),
                    "sha256": receipt["sha256"],
                }
            ).decode("ascii")
        )
        return 0
    campaigns = _selected_campaigns(args.campaign)
    results: list[dict[str, Any]] = []
    if args.action == "materialize":
        for campaign in campaigns:
            paths = materialize_campaign(
                campaign,
                output_root=args.output_root.resolve(),
                authorized_utc=args.authorized_utc,
            )
            results.append(
                {
                    "campaign": campaign.key,
                    "status": "materialized",
                    "paths": {
                        key: value.as_posix()
                        for key, value in paths.items()
                    },
                }
            )
    elif args.action == "validate":
        for campaign in campaigns:
            results.append(
                validate_campaign(
                    campaign,
                    output_root=args.output_root.resolve(),
                )
            )
    elif args.action == "release":
        for campaign in campaigns:
            paths = materialize_release_activation(
                campaign,
                output_root=args.output_root.resolve(),
                retirement_receipt_path=(
                    args.retirement_receipt.resolve()
                ),
                released_utc=args.released_utc,
            )
            results.append(
                {
                    "campaign": campaign.key,
                    "status": "released_activation_materialized",
                    "paths": {
                        key: value.as_posix()
                        for key, value in paths.items()
                    },
                }
            )
    elif args.action == "validate-release":
        for campaign in campaigns:
            results.append(
                validate_release_activation(
                    campaign,
                    output_root=args.output_root.resolve(),
                    retirement_receipt_path=(
                        args.retirement_receipt.resolve()
                    ),
                )
            )
    elif args.action == "release-held-v2":
        for campaign in campaigns:
            paths = materialize_ordinary_held_release_activation(
                campaign,
                output_root=args.output_root.resolve(),
                retirement_receipt_path=(
                    args.retirement_receipt.resolve()
                ),
                released_utc=args.released_utc,
            )
            results.append(
                {
                    "campaign": campaign.key,
                    "status": (
                        "ordinary_held_release_v2_materialized"
                    ),
                    "paths": {
                        key: value.as_posix()
                        for key, value in paths.items()
                    },
                }
            )
    else:
        for campaign in campaigns:
            results.append(
                validate_ordinary_held_release_activation(
                    campaign,
                    output_root=args.output_root.resolve(),
                    retirement_receipt_path=(
                        args.retirement_receipt.resolve()
                    ),
                )
            )
    print(canonical_json_bytes({"results": results}).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
