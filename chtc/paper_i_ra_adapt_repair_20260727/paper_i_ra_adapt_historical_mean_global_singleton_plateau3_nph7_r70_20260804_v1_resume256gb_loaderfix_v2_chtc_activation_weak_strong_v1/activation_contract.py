#!/usr/bin/env python3
"""One-row weak--strong staging activation for the exact-prefix r70 package."""

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
    "r70_20260804_v1_resume256gb_loaderfix_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_exact_prefix_49_45_31_resume256gb_loaderfix_v2"
)
ACTIVATION_ID = f"{PACKAGE_ID}_activation_weak_strong_v1"
ACTIVATION_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / ACTIVATION_ID
PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / PACKAGE_ID
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
LOADER_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_loaderfix_v2_chtc"
)
RUNTIME_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727"
) / f"{PACKAGE_ID}_runtime"
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
WEAK_EXECUTION_ID = (
    "historical_mean_global_singleton_v2_nph7_r70__weak_strong__nph7__"
    "ra_global_singleton_plateau__resume_from_d49_to_r70_256gb_loaderfix_v2"
)
BATCH_NAME = (
    "paper-i-ra-global-singleton-plateau-nph7-r70-exact49-256gb-weak-v1"
)
STAGING_ROOT = (
    "/staging/j/jsstrobel/"
    "paper_i_ra_historical_mean_global_singleton_r70_20260804_v1"
)
STAGING_OSDF_ROOT = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_historical_mean_global_singleton_r70_20260804_v1"
)
WEAK_ARCHIVE_BASENAME = "9401106.0__weak_strong__20260804T012503Z.tar.gz"
WEAK_ARCHIVE_SHA256 = (
    "c0589600744902f276c479fa05d7de53b55345b11221bd544de5183d8eabaf9c"
)
WEAK_ARCHIVE_SIZE_BYTES = 4_903_485_221
WEAK_INPUT_URI = f"{STAGING_OSDF_ROOT}/inputs/{WEAK_ARCHIVE_BASENAME}"
OUTPUT_DESTINATION = f"{STAGING_OSDF_ROOT}/outputs/"
OUTPUT_URI_TEMPLATE = (
    f"{STAGING_OSDF_ROOT}/outputs/"
    "attempt__<execution_id>__cluster_<cluster_id>__proc_<proc_id>__"
    "attempt_<attempt_ordinal>.tar.gz"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "resume256gb_execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "weak_staging_activation_manifest_v1"
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
    "resume_input_uri",
    "resume_archive_basename",
    "resume_archive_sha256",
    "cpus",
    "memory_mb",
    "disk_mb",
    "max_runtime_seconds",
)


class ActivationContractError(ValueError):
    """Raised when staging activation state is stale or malformed."""


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
        raise ActivationContractError(f"{label} must be an object.")
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256(value)
    if value.get("sha256") != observed:
        raise ActivationContractError(f"{label} digest drifted.")
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


def validate_package(repo_root: Path) -> dict[str, Any]:
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
            f"Sealed package validation failed: {completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ActivationContractError(
            "Package validator returned malformed JSON."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status")
        != "passed_inert_three_authenticated_r70_resumes"
        or payload.get("row_count") != 3
        or payload.get("source_preflight_count") != 3
        or payload.get("resume_controller_rounds") != [49, 45, 31]
        or payload.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or payload.get("non_swept_settings_diff") != []
        or payload.get("resources", {}).get("request_cpus") != 4
        or payload.get("resources", {}).get("request_memory_mb") != 262_144
        or payload.get("resources", {}).get("request_disk_mb") != 102_400
        or payload.get("source_held_jobs_preserved") is not True
        or payload.get("submitted") is not False
    ):
        raise ActivationContractError("Package validation closure drifted.")
    return payload


def weak_job(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    package = repo_root / PACKAGE_RELATIVE
    manifest = load_json(package / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    matches = [
        row
        for row in manifest.get("jobs", [])
        if isinstance(row, Mapping)
        and row.get("execution_id") == WEAK_EXECUTION_ID
    ]
    if len(matches) != 1:
        raise ActivationContractError("Weak job binding is not unique.")
    binding = matches[0]
    path = package / safe_relative_path(binding.get("path"), label="job path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ActivationContractError("Weak job bytes drifted.")
    job = load_json(path, label="weak job")
    if (
        verify_self_digest(job, label="weak job")
        != binding.get("canonical_sha256")
        or job.get("execution_id") != WEAK_EXECUTION_ID
        or job.get("regime_id") != "weak_strong"
        or job.get("resume_input", {}).get("resume_controller_round") != 49
        or job.get("resume_input", {}).get("runtime_archive_basename")
        != WEAK_ARCHIVE_BASENAME
        or job.get("resume_input", {}).get("local_archive", {}).get("sha256")
        != WEAK_ARCHIVE_SHA256
        or job.get("resources", {}).get("request_memory_mb") != 262_144
        or job.get("resources", {}).get("request_disk_mb") != 102_400
    ):
        raise ActivationContractError("Weak job scientific identity drifted.")
    return path, job


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
        "job_sha256": job["sha256"],
        "package_manifest_sha256": package_manifest["sha256"],
        "derived_protocol_sha256": job["derived_protocol_sha256"],
        "checkpoint_sha256": resume["checkpoint_sha256"],
        "resume_controller_round": resume["resume_controller_round"],
        "target_horizon": 70,
        "resources": job["resources"],
        "authorization_kind": (
            "explicit_user_execution_and_submission_authority"
        ),
        "authorization_source": "explicit_user_request_2026-08-04",
        "authorized_utc": authorized_utc,
        "scope": "weak_strong_exact_prefix_only",
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
    return f"""# Weak--strong exact-prefix k49 to r70, staging-backed and unheld.
universe = vanilla
executable = /bin/bash
transfer_executable = False

arguments = {activation}/execute_resume_job.sh {activation} {package} $(job_path) $(job_file_sha256) $(authorization_path) $(authorization_file_sha256) {SOURCE_ARCHIVE_PATH} {SOURCE_ARCHIVE_SHA256} {IMAGE_PATH} {IMAGE_SHA256} $(resume_archive_basename) $(resume_archive_sha256) attempt

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {SOURCE_PACKAGE_RELATIVE.as_posix()}, {LOADER_PACKAGE_RELATIVE.as_posix()}, {package}, {activation}, {IMAGE_PATH}, $(resume_input_uri)
output_destination = {OUTPUT_DESTINATION}

request_cpus = $(cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
requirements = (TARGET.HasSIF == true) && (TARGET.HasCHTCStaging == true)

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
    required = (
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        f"$(resume_input_uri)",
        f"output_destination = {OUTPUT_DESTINATION}",
        "TARGET.HasCHTCStaging == true",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        f'+JobBatchName = "{BATCH_NAME}"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "periodic_release = False",
        "leave_in_queue = False",
        "queue " + ",".join(QUEUE_VARIABLES) + " from ",
    )
    if any(token not in text for token in required):
        raise ActivationContractError("Submit descriptor contract drifted.")
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "max_materialize",
            "max_idle",
            "condor_hold",
            "condor_release",
            "condor_rm",
        )
    ):
        raise ActivationContractError("Forbidden lifecycle control present.")
    if re.search(r"(?im)^\s*hold\s*=", text):
        raise ActivationContractError("Job must not submit held.")
    if "transfer_output_remaps" in lowered:
        raise ActivationContractError(
            "Attempt-safe output_destination must not be mixed with remaps."
        )
