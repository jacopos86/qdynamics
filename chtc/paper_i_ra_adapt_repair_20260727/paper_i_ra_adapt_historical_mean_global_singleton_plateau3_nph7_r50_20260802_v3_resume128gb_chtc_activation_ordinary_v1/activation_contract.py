#!/usr/bin/env python3
"""Activation contract for the three 128-GiB accepted-state resumes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_resume128gb_v3"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_ordinary_v1"
ACTIVATION_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{ACTIVATION_ID}"
)
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{PACKAGE_ID}"
)
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
RUNTIME_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    f"{PACKAGE_ID}_runtime"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
SOURCE_ARCHIVE_PATH = (
    SOURCE_PACKAGE_RELATIVE / "source/source_locked.tar.gz"
).as_posix()
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)
BATCH_NAME = (
    "paper-i-ra-global-singleton-plateau-nph7-r50-resume128gb-v3"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "activation_manifest_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
    "execution_authorization_v1"
)
CONTROL_FILES = (
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "execute_resume_job.sh",
    "build_attempt_archive.py",
)
GENERATED_PATHS = (
    "authorizations",
    "queue.tsv",
    "submit.sub",
    "activation_manifest.json",
)
QUEUE_VARIABLES = (
    "execution_id",
    "job_path",
    "job_file_sha256",
    "authorization_path",
    "authorization_file_sha256",
    "resume_archive_path",
    "resume_archive_sha256",
    "cpus",
    "memory_mb",
    "disk_mb",
    "max_runtime_seconds",
)


class ActivationContractError(ValueError):
    """Raised when activation state is stale or malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivationContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ActivationContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256(value)
    if value.get("sha256") != observed:
        raise ActivationContractError(f"{label} canonical digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ActivationContractError(f"{label} must be a relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(
        part in {"", ".", ".."} for part in pure.parts
    ):
        raise ActivationContractError(f"Unsafe {label}: {value}")
    return Path(*pure.parts)


def repo_root_from_script(path: str | Path) -> Path:
    for candidate in Path(path).resolve().parents:
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines"
        ).is_dir():
            return candidate
    raise ActivationContractError("Active repository root was not found.")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError(f"Unsafe bound file: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=path.name)
    canonical = verify_self_digest(payload, label=path.name)
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": canonical,
    }


def authorization_payload(
    *,
    job: Mapping[str, Any],
    package_manifest: Mapping[str, Any],
    control_sha256: str,
    authorized_utc: str,
) -> dict[str, Any]:
    resume = job["resume_input"]
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "passed",
        "activation_id": ACTIVATION_ID,
        "activation_control_plane_sha256": control_sha256,
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": job["execution_id"],
        "source_execution_id": job["source_execution_id"],
        "source_cluster_id": job["source_cluster_id"],
        "source_proc_id": job["source_proc_id"],
        "job_sha256": job["sha256"],
        "package_manifest_sha256": package_manifest["sha256"],
        "scientific_protocol_sha256": job["scientific_protocol_sha256"],
        "checkpoint_sha256": resume["checkpoint_sha256"],
        "resume_controller_round": resume["resume_controller_round"],
        "target_horizon": job["target_horizon"],
        "resources": job["resources"],
        "authorization_kind": (
            "explicit_user_execution_and_submission_authority"
        ),
        "authorization_source": "explicit_user_request_2026-08-02",
        "authorized_utc": authorized_utc,
        "scope": "three_held_memory_repair_resumes_exact_cell_only",
        "execution_authorized": True,
        "submission_authorized": True,
        "source_held_job_removal_authorized": False,
        "paper_evidence_adoption_authorized": False,
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }


def render_submit() -> str:
    activation = ACTIVATION_RELATIVE.as_posix()
    package = PACKAGE_RELATIVE.as_posix()
    runtime = RUNTIME_RELATIVE.as_posix()
    queue_columns = ",".join(QUEUE_VARIABLES)
    return f"""# Three source-exact nph7 accepted-state memory-repair resumes.
universe = vanilla
executable = /bin/bash
transfer_executable = False

arguments = {activation}/execute_resume_job.sh {activation} {package} $(job_path) $(job_file_sha256) $(authorization_path) $(authorization_file_sha256) {SOURCE_ARCHIVE_PATH} {SOURCE_ARCHIVE_SHA256} {IMAGE_PATH} {IMAGE_SHA256} $(resume_archive_path) $(resume_archive_sha256) transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {SOURCE_PACKAGE_RELATIVE.as_posix()}, {package}, {activation}, {IMAGE_PATH}, $(resume_archive_path)
transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz
transfer_output_remaps = \"transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz={runtime}/fetched/$(execution_id)__cluster_$(ClusterId)__proc_$(ProcId).tar.gz\"

request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
requirements = TARGET.HasSIF

notification = Never
getenv = False
stream_output = False
stream_error = False
+JobBatchName = \"{BATCH_NAME}\"
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
periodic_release = False
leave_in_queue = False
kill_sig = SIGTERM
kill_sig_timeout = 600

log = {runtime}/logs/$(Cluster).$(Process)__$(execution_id).log
output = {runtime}/logs/$(Cluster).$(Process)__$(execution_id).out
error = {runtime}/logs/$(Cluster).$(Process)__$(execution_id).err

queue {queue_columns} from {activation}/queue.tsv
"""


def validate_submit_text(text: str) -> None:
    """Validate the exact ordinary, unheld, row-specific transfer lifecycle."""

    required = (
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "preserve_relative_paths = True",
        "$(resume_archive_path)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        f'+JobBatchName = "{BATCH_NAME}"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "periodic_release = False",
        "leave_in_queue = False",
        "kill_sig = SIGTERM",
        "kill_sig_timeout = 600",
        "queue " + ",".join(QUEUE_VARIABLES) + " from ",
    )
    if any(token not in text for token in required):
        raise ActivationContractError("Submit descriptor contract drifted.")
    lowered = text.lower()
    forbidden = (
        "max_materialize",
        "max_idle",
        "condor_hold",
        "condor_release",
        "condor_rm",
        "condor_remove",
    )
    if any(token in lowered for token in forbidden):
        raise ActivationContractError(
            "Submit descriptor contains a forbidden lifecycle control."
        )
    if re.search(r"(?im)^\s*hold\s*=", text):
        raise ActivationContractError("Resume jobs must not submit held.")
    if len(re.findall(r"(?im)^\s*leave_in_queue\s*=", text)) != 1:
        raise ActivationContractError("Lifecycle assignment is not unique.")
    transfer_inputs = re.findall(
        r"(?im)^\s*transfer_input_files\s*=\s*(.*?)\s*$", text
    )
    if len(transfer_inputs) != 1 or transfer_inputs[0].count(
        "$(resume_archive_path)"
    ) != 1:
        raise ActivationContractError(
            "Each row must transfer exactly its own resume archive."
        )


def _run_package_validator(repo_root: Path) -> dict[str, Any]:
    validator = repo_root / PACKAGE_RELATIVE / "validate_package.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(validator),
            "--metadata-only",
            "--source-preflight",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise ActivationContractError(
            f"Resume package validation failed: {completed.stderr}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ActivationContractError(
            "Resume package validator returned malformed JSON."
        ) from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("status")
        != "passed_inert_three_authenticated_resumes"
        or receipt.get("row_count") != 3
        or receipt.get("source_preflight_count") != 3
        or receipt.get("scientific_protocol_changed") is not False
        or receipt.get("scientific_settings_changed") != []
        or receipt.get("source_held_jobs_preserved") is not True
        or receipt.get("request_memory_mb") != 131_072
        or receipt.get("request_disk_mb") != 81_920
        or receipt.get("submitted") is not False
    ):
        raise ActivationContractError("Resume package validator drifted.")
    return receipt


def _parse_queue(path: Path) -> list[list[str]]:
    if not path.is_file() or path.is_symlink():
        raise ActivationContractError("Activation queue is unsafe.")
    rows = [line.split("\t") for line in path.read_text().splitlines()]
    if len(rows) != 3 or any(len(row) != len(QUEUE_VARIABLES) for row in rows):
        raise ActivationContractError("Activation queue shape drifted.")
    return rows


def validate_activation(repo_root: str | Path) -> dict[str, Any]:
    """Validate the complete authorized-but-unsubmitted activation closure."""

    root = Path(repo_root).resolve()
    activation_dir = root / ACTIVATION_RELATIVE
    package_dir = root / PACKAGE_RELATIVE
    if Path(__file__).resolve().parent != activation_dir:
        raise ActivationContractError("Activation directory identity drifted.")
    package_validation = _run_package_validator(root)
    package_manifest_path = package_dir / "package_manifest.json"
    package_manifest = load_json(
        package_manifest_path, label="resume package manifest"
    )
    verify_self_digest(package_manifest, label="resume package manifest")
    if (
        package_manifest.get("package_id") != PACKAGE_ID
        or package_manifest.get("campaign_id") != CAMPAIGN_ID
        or package_manifest.get("sha256")
        != package_validation.get("package_manifest_sha256")
        or package_manifest.get("row_count") != 3
        or package_manifest.get("execution_authorized") is not False
        or package_manifest.get("submission_authorized") is not False
        or package_manifest.get("submitted") is not False
    ):
        raise ActivationContractError("Sealed resume package authority drifted.")

    controls = [
        file_binding(activation_dir / name, relative_to=activation_dir)
        for name in CONTROL_FILES
    ]
    control_sha = canonical_sha256({"controls": controls})
    submit_path = activation_dir / "submit.sub"
    submit_text = submit_path.read_text(encoding="utf-8")
    validate_submit_text(submit_text)
    if submit_text != render_submit():
        raise ActivationContractError("Rendered submit descriptor drifted.")

    raw_job_bindings = package_manifest.get("jobs")
    if not isinstance(raw_job_bindings, list) or len(raw_job_bindings) != 3:
        raise ActivationContractError("Resume package job closure drifted.")
    jobs: list[tuple[Path, dict[str, Any]]] = []
    for binding in raw_job_bindings:
        if not isinstance(binding, Mapping):
            raise ActivationContractError("Resume package job binding drifted.")
        job_path = package_dir / safe_relative_path(
            binding.get("path"), label="resume job path"
        )
        job = load_json(job_path, label="resume job")
        if (
            verify_self_digest(job, label="resume job")
            != binding.get("canonical_sha256")
            or sha256_file(job_path) != binding.get("sha256")
            or job_path.stat().st_size != binding.get("size_bytes")
        ):
            raise ActivationContractError("Resume job binding drifted.")
        jobs.append((job_path, job))

    authorizations: list[dict[str, Any]] = []
    authorization_paths: dict[str, Path] = {}
    authorized_utc: str | None = None
    for job_path, job in jobs:
        identifier = str(job["execution_id"])
        path = activation_dir / "authorizations" / f"{identifier}.json"
        authorization = load_json(path, label="execution authorization")
        verify_self_digest(authorization, label="execution authorization")
        row_utc = authorization.get("authorized_utc")
        if not isinstance(row_utc, str) or not row_utc.strip():
            raise ActivationContractError("Authorization timestamp is absent.")
        expected = authorization_payload(
            job=job,
            package_manifest=package_manifest,
            control_sha256=control_sha,
            authorized_utc=row_utc,
        )
        if set(authorization) != set(expected) | {"sha256"}:
            raise ActivationContractError(
                f"Authorization field closure drifted: {identifier}."
            )
        if any(authorization.get(key) != value for key, value in expected.items()):
            raise ActivationContractError(
                f"Authorization semantics drifted: {identifier}."
            )
        if authorized_utc is None:
            authorized_utc = row_utc
        elif authorized_utc != row_utc:
            raise ActivationContractError("Authorization timestamps diverged.")
        authorization_paths[identifier] = path
        authorizations.append(
            {"execution_id": identifier, **json_binding(path, relative_to=activation_dir)}
        )
    assert authorized_utc is not None

    queue_path = activation_dir / "queue.tsv"
    queue_rows = _parse_queue(queue_path)
    executions: list[dict[str, Any]] = []
    for index, (queue_row, (job_path, job)) in enumerate(
        zip(queue_rows, jobs, strict=True)
    ):
        identifier = str(job["execution_id"])
        authorization_path = authorization_paths[identifier]
        resume = job["resume_input"]["archive"]
        resources = job["resources"]
        expected_row = [
            identifier,
            job_path.relative_to(root).as_posix(),
            sha256_file(job_path),
            authorization_path.relative_to(root).as_posix(),
            sha256_file(authorization_path),
            str(resume["path"]),
            str(resume["sha256"]),
            str(resources["request_cpus"]),
            str(resources["request_memory_mb"]),
            str(resources["request_disk_mb"]),
            str(resources["max_runtime_seconds"]),
        ]
        if queue_row != expected_row:
            raise ActivationContractError(f"Activation queue row {index} drifted.")
        if (
            resources.get("request_memory_mb") != 131_072
            or resources.get("request_disk_mb") != 81_920
            or job.get("scientific_protocol_changed") is not False
            or job.get("scientific_settings_changed") != []
            or job.get("source_job_preserved_held") is not True
        ):
            raise ActivationContractError(f"Resume job {identifier} drifted.")
        executions.append(
            {
                "queue_index": index,
                "execution_id": identifier,
                "source_cluster_id": job["source_cluster_id"],
                "source_proc_id": job["source_proc_id"],
                "source_held_job_preserved": True,
                "job": json_binding(job_path, relative_to=root),
                "authorization": json_binding(
                    authorization_path, relative_to=root
                ),
                "resume_archive": resume,
                "resources": resources,
            }
        )

    manifest_path = activation_dir / "activation_manifest.json"
    manifest = load_json(manifest_path, label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    expected_manifest = {
        "schema": ACTIVATION_SCHEMA,
        "status": "passed_authorized_not_submitted",
        "activation_id": ACTIVATION_ID,
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "batch_name": BATCH_NAME,
        "run_class": "diagnostic",
        "execution_target": "chtc",
        "authorized_utc": authorized_utc,
        "direct_execution_count": 3,
        "sealed_package": {
            "path": PACKAGE_RELATIVE.as_posix(),
            "manifest": json_binding(package_manifest_path, relative_to=root),
            "validator_status": package_validation["status"],
            "source_preflight_count": package_validation[
                "source_preflight_count"
            ],
        },
        "source_package_path": SOURCE_PACKAGE_RELATIVE.as_posix(),
        "remote_image": {
            "path": IMAGE_PATH,
            "sha256": IMAGE_SHA256,
            "byte_verification_required_before_submit": True,
            "byte_verification_passed": False,
        },
        "control_plane": controls,
        "activation_control_plane_sha256": control_sha,
        "execution_authorizations": authorizations,
        "executions": executions,
        "queue": file_binding(queue_path, relative_to=activation_dir),
        "submit_descriptor": file_binding(submit_path, relative_to=activation_dir),
        "queue_variables": list(QUEUE_VARIABLES),
        "runtime_root": RUNTIME_RELATIVE.as_posix(),
        "ordinary_cluster": True,
        "bounded_factory": False,
        "source_held_jobs_preserved": True,
        "source_held_job_removal_authorized": False,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
        "pre_submit_requirements": [
            "verify_remote_image_exact_path_size_and_sha256",
            "verify_three_remote_resume_archives_exact_size_and_sha256",
            "validate_exact_submit_descriptor_lifecycle",
            "condor_submit_dry_run_exact_descriptor",
            "check_exact_batch_and_execution_id_collisions",
            "check_home_quota_available",
        ],
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_pending_remote_preflight",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
        "paper_evidence_adopted": False,
    }
    if set(manifest) != set(expected_manifest) | {"sha256"}:
        raise ActivationContractError("Activation manifest field closure drifted.")
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ActivationContractError("Activation manifest semantics drifted.")
    return {
        "status": "passed_authorized_not_submitted",
        "activation_id": ACTIVATION_ID,
        "activation_manifest_sha256": manifest["sha256"],
        "package_manifest_sha256": package_manifest["sha256"],
        "direct_execution_count": 3,
        "source_preflight_count": 3,
        "resume_controller_rounds": [
            job["resume_input"]["resume_controller_round"] for _path, job in jobs
        ],
        "request_memory_mb": 131_072,
        "request_disk_mb": 81_920,
        "row_specific_resume_archives": True,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
        "source_held_jobs_preserved": True,
        "source_held_job_removal_authorized": False,
        "execution_authorized": True,
        "submission_authorized": True,
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
