#!/usr/bin/env python3
"""Materialize the two missing strong-Holstein r70 resume activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


ACTIVATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = ACTIVATION_DIR.parents[2]
BASE_RELATIVE = Path("chtc/paper_i_ra_adapt_repair_20260727")
PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_20260804_v1_resume256gb_loaderfix_v2_chtc"
)
PACKAGE_RELATIVE = BASE_RELATIVE / PACKAGE_ID
PACKAGE_DIR = REPO_ROOT / PACKAGE_RELATIVE
WEAK_ACTIVATION_RELATIVE = BASE_RELATIVE / f"{PACKAGE_ID}_activation_weak_strong_v1"
WEAK_ACTIVATION_DIR = REPO_ROOT / WEAK_ACTIVATION_RELATIVE
ACTIVATION_ID = f"{PACKAGE_ID}_activation_remaining_strong2_20260806_v1"
ACTIVATION_RELATIVE = BASE_RELATIVE / ACTIVATION_ID
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r70_exact_prefix_49_45_31_resume256gb_loaderfix_v2"
)
BATCH_NAME = (
    "paper-i-ra-global-singleton-plateau-nph7-r70-"
    "exact45-31-256gb-remaining2-v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "resume256gb_execution_authorization_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_r70_"
    "remaining_strong2_activation_v1"
)
SOURCE_PACKAGE_RELATIVE = BASE_RELATIVE / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
LOADER_PACKAGE_RELATIVE = BASE_RELATIVE / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_loaderfix_v2_chtc"
)
SOURCE_ARCHIVE_PATH = (
    SOURCE_PACKAGE_RELATIVE / "source/source_locked.tar.gz"
).as_posix()
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
STAGING_ROOT = (
    "/staging/j/jsstrobel/"
    "paper_i_ra_historical_mean_global_singleton_r70_20260804_v1"
)
STAGING_OSDF_ROOT = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_historical_mean_global_singleton_r70_20260804_v1"
)
OUTPUT_DESTINATION = f"{STAGING_OSDF_ROOT}/outputs/"
RUNTIME_RELATIVE = BASE_RELATIVE / f"{PACKAGE_ID}_runtime"
EXECUTION_IDS = (
    "historical_mean_global_singleton_v2_nph7_r70__intermediate_strong__"
    "nph7__ra_global_singleton_plateau__resume_from_d45_to_r70_256gb_loaderfix_v2",
    "historical_mean_global_singleton_v2_nph7_r70__strong_strong_u8__"
    "nph7__ra_global_singleton_plateau__resume_from_d31_to_r70_256gb_loaderfix_v2",
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
CONTROL_FILES = (
    "materialize_activation.py",
    "execute_resume_job.sh",
    "build_attempt_archive.py",
)
GENERATED_PATHS = (
    "execute_resume_job.sh",
    "build_attempt_archive.py",
    "authorizations",
    "queue.tsv",
    "submit.sub",
    "activation_manifest.json",
)


class ActivationError(ValueError):
    """Raised when activation bytes or bindings drift."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


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
    if not path.is_file() or path.is_symlink():
        raise ActivationError(f"Unsafe {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ActivationError(f"{label} must be an object")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("sha256") != canonical_sha256(payload):
        raise ActivationError(f"{label} canonical digest drifted")


def file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ActivationError(f"Unsafe bound file: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = load_json(path, label=path.name)
    verify_self_digest(payload, label=path.name)
    return {
        **file_binding(path, relative_to=relative_to),
        "canonical_sha256": payload["sha256"],
    }


def exclusive_write(path: Path, data: bytes, *, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ActivationError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        created.append(path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Mapping[str, Any], *, created: list[Path]) -> None:
    exclusive_write(path, canonical_json_bytes(payload) + b"\n", created=created)


def load_package() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "validate_package.py"),
            "--metadata-only",
            "--source-preflight",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise ActivationError(f"Package validation failed: {completed.stderr[-1000:]}")
    receipt = json.loads(completed.stdout)
    if (
        receipt.get("status") != "passed_inert_three_authenticated_r70_resumes"
        or receipt.get("row_count") != 3
        or receipt.get("source_preflight_count") != 3
        or receipt.get("resume_controller_rounds") != [49, 45, 31]
        or receipt.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or receipt.get("non_swept_settings_diff") != []
    ):
        raise ActivationError("Sealed package validation closure drifted")
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    jobs: dict[str, dict[str, Any]] = {}
    for binding in manifest.get("jobs", []):
        execution_id = binding.get("execution_id")
        if execution_id not in EXECUTION_IDS:
            continue
        path = PACKAGE_DIR / binding["path"]
        if (
            path.stat().st_size != binding["size_bytes"]
            or sha256_file(path) != binding["sha256"]
        ):
            raise ActivationError(f"Job file binding drifted: {execution_id}")
        job = load_json(path, label=f"job {execution_id}")
        verify_self_digest(job, label=f"job {execution_id}")
        if job["sha256"] != binding["canonical_sha256"]:
            raise ActivationError(f"Job canonical binding drifted: {execution_id}")
        jobs[execution_id] = {"path": path, "job": job}
    if tuple(jobs) != EXECUTION_IDS:
        raise ActivationError("The two requested jobs are not uniquely bound")
    return manifest, jobs


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
        "authorization_kind": "explicit_user_execution_and_submission_authority",
        "authorization_source": "explicit_user_request_2026-08-06",
        "scope": "weak_strong_exact_prefix_only",
        "scope_compatibility_note": (
            "The sealed v1 runner requires this legacy token for every package row; "
            "the effective scope is bound by execution_id, job digest, regime, "
            "checkpoint digest, and protocol digest."
        ),
        "effective_regime_scope": job["regime_id"],
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
    return f"""# Missing intermediate--strong and strong--strong exact-prefix resumes to r70.
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


def materialize(*, authorized_utc: str) -> dict[str, Any]:
    if ACTIVATION_DIR != REPO_ROOT / ACTIVATION_RELATIVE:
        raise ActivationError("Activation path identity drifted")
    if not authorized_utc.strip():
        raise ActivationError("Authorization timestamp is required")
    for name in GENERATED_PATHS:
        if (ACTIVATION_DIR / name).exists() or (ACTIVATION_DIR / name).is_symlink():
            raise ActivationError(f"Refusing to overwrite generated path: {name}")
    created: list[Path] = []
    try:
        for name in ("execute_resume_job.sh", "build_attempt_archive.py"):
            source = WEAK_ACTIVATION_DIR / name
            destination = ACTIVATION_DIR / name
            exclusive_write(destination, source.read_bytes(), created=created)
            destination.chmod(source.stat().st_mode & 0o777)
        package_manifest, jobs = load_package()
        controls = [
            file_binding(ACTIVATION_DIR / name, relative_to=ACTIVATION_DIR)
            for name in CONTROL_FILES
        ]
        control_sha = canonical_sha256({"controls": controls})
        queue_lines: list[str] = []
        authorizations: list[dict[str, Any]] = []
        executions: list[dict[str, Any]] = []
        for execution_id in EXECUTION_IDS:
            row = jobs[execution_id]
            job_path = row["path"]
            job = row["job"]
            resume = job["resume_input"]
            authorization = digested(
                authorization_payload(
                    job=job,
                    package_manifest=package_manifest,
                    control_sha256=control_sha,
                    authorized_utc=authorized_utc,
                )
            )
            authorization_path = ACTIVATION_DIR / "authorizations" / f"{execution_id}.json"
            write_json(authorization_path, authorization, created=created)
            archive = resume["local_archive"]
            archive_basename = resume["runtime_archive_basename"]
            input_uri = f"{STAGING_OSDF_ROOT}/inputs/{archive_basename}"
            resources = job["resources"]
            queue_lines.append(
                "\t".join(
                    (
                        execution_id,
                        job_path.relative_to(REPO_ROOT).as_posix(),
                        sha256_file(job_path),
                        authorization_path.relative_to(REPO_ROOT).as_posix(),
                        sha256_file(authorization_path),
                        input_uri,
                        archive_basename,
                        archive["sha256"],
                        str(resources["request_cpus"]),
                        str(resources["request_memory_mb"]),
                        str(resources["request_disk_mb"]),
                        str(resources["max_runtime_seconds"]),
                    )
                )
                + "\n"
            )
            authorizations.append(
                {"execution_id": execution_id, **json_binding(authorization_path, relative_to=REPO_ROOT)}
            )
            executions.append(
                {
                    "execution_id": execution_id,
                    "regime_id": job["regime_id"],
                    "resume_controller_round": resume["resume_controller_round"],
                    "target_horizon": 70,
                    "job": json_binding(job_path, relative_to=REPO_ROOT),
                    "authorization": json_binding(authorization_path, relative_to=REPO_ROOT),
                    "resume_archive": archive,
                    "resume_input_uri": input_uri,
                    "resources": resources,
                }
            )
        queue_path = ACTIVATION_DIR / "queue.tsv"
        exclusive_write(queue_path, "".join(queue_lines).encode("utf-8"), created=created)
        submit_path = ACTIVATION_DIR / "submit.sub"
        exclusive_write(submit_path, render_submit().encode("utf-8"), created=created)
        manifest = digested(
            {
                "schema": ACTIVATION_SCHEMA,
                "status": "passed_authorized_pending_staging",
                "activation_id": ACTIVATION_ID,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "batch_name": BATCH_NAME,
                "run_class": "diagnostic",
                "execution_target": "chtc",
                "authorized_utc": authorized_utc,
                "direct_execution_count": 2,
                "activation_execution_ids": list(EXECUTION_IDS),
                "activation_policy_override": {
                    "prior_cardinality": 1,
                    "authorized_cardinality": 2,
                    "reason": "explicit_user_request_to_submit_both_missing_rows_2026-08-06",
                    "scientific_settings_changed": [],
                },
                "only_scientific_change": "maximum_controller_rounds_50_to_70",
                "non_swept_settings_diff": [],
                "sealed_package": {
                    "path": PACKAGE_RELATIVE.as_posix(),
                    "manifest": json_binding(PACKAGE_DIR / "package_manifest.json", relative_to=REPO_ROOT),
                    "validator_status": "passed_inert_three_authenticated_r70_resumes",
                },
                "control_plane": controls,
                "activation_control_plane_sha256": control_sha,
                "execution_authorizations": authorizations,
                "executions": executions,
                "queue": file_binding(queue_path, relative_to=ACTIVATION_DIR),
                "submit_descriptor": file_binding(submit_path, relative_to=ACTIVATION_DIR),
                "queue_variables": list(QUEUE_VARIABLES),
                "remote_image": {
                    "path": IMAGE_PATH,
                    "sha256": IMAGE_SHA256,
                    "byte_verification_required_before_submit": True,
                },
                "staging": {
                    "path": STAGING_ROOT,
                    "expected_quota_gb": 100,
                    "expected_item_limit": 1000,
                    "output_destination": OUTPUT_DESTINATION,
                },
                "ordinary_cluster": True,
                "bounded_factory": False,
                "intentional_hold": False,
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_pending_staging",
                "submitted": False,
                "paper_evidence_adopted": False,
            }
        )
        write_json(ACTIVATION_DIR / "activation_manifest.json", manifest, created=created)
        return validate()
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        auth_dir = ACTIVATION_DIR / "authorizations"
        if auth_dir.is_dir() and not auth_dir.is_symlink():
            try:
                auth_dir.rmdir()
            except OSError:
                pass
        raise


def validate() -> dict[str, Any]:
    package_manifest, jobs = load_package()
    manifest = load_json(ACTIVATION_DIR / "activation_manifest.json", label="activation manifest")
    verify_self_digest(manifest, label="activation manifest")
    if (
        manifest.get("activation_id") != ACTIVATION_ID
        or manifest.get("batch_name") != BATCH_NAME
        or manifest.get("direct_execution_count") != 2
        or manifest.get("activation_execution_ids") != list(EXECUTION_IDS)
        or manifest.get("execution_authorized") is not True
        or manifest.get("submission_authorized") is not True
        or manifest.get("intentional_hold") is not False
        or manifest.get("bounded_factory") is not False
        or manifest.get("submitted") is not False
    ):
        raise ActivationError("Activation manifest closure drifted")
    if (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8") != render_submit():
        raise ActivationError("Submit descriptor drifted")
    completed = subprocess.run(
        ["bash", "-n", str(ACTIVATION_DIR / "execute_resume_job.sh")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ActivationError(f"Worker shell syntax failed: {completed.stderr}")
    rows = (ACTIVATION_DIR / "queue.tsv").read_text(encoding="utf-8").splitlines()
    if len(rows) != 2:
        raise ActivationError("Activation queue must contain exactly two rows")
    observed_ids = tuple(row.split("\t", 1)[0] for row in rows)
    if observed_ids != EXECUTION_IDS:
        raise ActivationError("Activation queue execution ordering drifted")
    for execution_id in EXECUTION_IDS:
        authorization = load_json(
            ACTIVATION_DIR / "authorizations" / f"{execution_id}.json",
            label=f"authorization {execution_id}",
        )
        verify_self_digest(authorization, label=f"authorization {execution_id}")
        expected = authorization_payload(
            job=jobs[execution_id]["job"],
            package_manifest=package_manifest,
            control_sha256=manifest["activation_control_plane_sha256"],
            authorized_utc=manifest["authorized_utc"],
        )
        if any(authorization.get(key) != value for key, value in expected.items()):
            raise ActivationError(f"Authorization drifted: {execution_id}")
    return {
        "status": "passed_authorized_pending_staging",
        "activation_id": ACTIVATION_ID,
        "batch_name": BATCH_NAME,
        "activation_manifest_sha256": manifest["sha256"],
        "direct_execution_count": 2,
        "execution_ids": list(EXECUTION_IDS),
        "resume_controller_rounds": [45, 31],
        "target_horizon": 70,
        "request_memory_mb": 262144,
        "request_disk_mb": 102400,
        "intentional_hold": False,
        "submitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--authorized-utc")
    group.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    payload = (
        validate()
        if args.validate
        else materialize(authorized_utc=args.authorized_utc)
    )
    print(canonical_json_bytes(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
