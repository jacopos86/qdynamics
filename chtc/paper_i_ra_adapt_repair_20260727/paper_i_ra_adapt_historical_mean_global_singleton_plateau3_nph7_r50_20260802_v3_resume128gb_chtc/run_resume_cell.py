#!/usr/bin/env python3
"""Execute one source-exact accepted-state memory-repair continuation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from types import ModuleType
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from resume_contract import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    SOURCE_PACKAGE_MANIFEST_FILE_SHA256,
    SOURCE_PACKAGE_RELATIVE,
    SOURCE_RUNNER_SHA256,
    TARGET_HORIZON,
    ResumeContractError,
    canonical_json_bytes,
    load_json,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


def _runtime_root() -> Path:
    root = PACKAGE_DIR
    for _part in PACKAGE_RELATIVE.parts:
        root = root.parent
    if root / PACKAGE_RELATIVE != PACKAGE_DIR:
        raise ResumeContractError("Runtime package path is not repo-relative.")
    return root


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumeContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResumeContractError(f"{label} must be a list.")
    return value


def _runtime_path(value: Any, *, label: str) -> Path:
    root = _runtime_root()
    path = root / safe_relative_path(value, label=label)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ResumeContractError(f"{label} escaped runtime root.") from exc
    return path


def _verify_file_binding(
    raw: Any, *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    binding = _mapping(raw, label=f"{label} binding")
    path = _runtime_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise ResumeContractError(f"{label} exact bytes drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise ResumeContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_runtime_job(job_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = _runtime_root()
    manifest_path = root / PACKAGE_RELATIVE / "package_manifest.json"
    manifest = load_json(manifest_path, label="resume package manifest")
    verify_self_digest(manifest, label="resume package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "passed_inert_three_authenticated_resumes"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("row_count") != 3
        or manifest.get("scientific_protocol_changed") is not False
        or manifest.get("scientific_settings_changed") != []
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
    ):
        raise ResumeContractError("Resume package manifest drifted.")
    resolved = job_path.resolve()
    bindings = _sequence(manifest.get("jobs"), label="resume job bindings")
    matches = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        candidate = root / PACKAGE_RELATIVE / safe_relative_path(
            binding.get("path"), label="resume job path"
        )
        if candidate.resolve() == resolved:
            matches.append(binding)
    if len(matches) != 1:
        raise ResumeContractError("Resume job is outside package closure.")
    binding = matches[0]
    if (
        not job_path.is_file()
        or job_path.is_symlink()
        or job_path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(job_path) != binding.get("sha256")
    ):
        raise ResumeContractError("Resume job byte binding drifted.")
    job = load_json(job_path, label="resume job")
    if verify_self_digest(job, label="resume job") != binding.get(
        "canonical_sha256"
    ):
        raise ResumeContractError("Resume job canonical binding drifted.")
    resume = _mapping(job.get("resume_input"), label="resume input")
    members = _sequence(resume.get("members"), label="resume members")
    roles = {str(row.get("role")) for row in members if isinstance(row, Mapping)}
    checkpoint_rows = [
        row
        for row in members
        if isinstance(row, Mapping) and row.get("role") == "checkpoint"
    ]
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_mode")
        != "authenticated_accepted_state_resume_to_50"
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("route_profile") != ROUTE_PROFILE
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("source_job_preserved_held") is not True
        or job.get("scientific_protocol_changed") is not False
        or job.get("scientific_settings_changed") != []
        or job.get("resources", {}).get("request_memory_mb") != 131_072
        or job.get("resources", {}).get("request_disk_mb") != 81_920
        or resume.get("pointer_closed") is not True
        or resume.get("validation_status") != "passed"
        or resume.get("member_count") != 3
        or len(members) != 3
        or roles
        != {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        or len(checkpoint_rows) != 1
        or checkpoint_rows[0].get("path") != resume.get("checkpoint_path")
        or checkpoint_rows[0].get("sha256")
        != resume.get("checkpoint_sha256")
        or not 0
        < int(resume.get("resume_controller_round", -1))
        < TARGET_HORIZON
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise ResumeContractError("Accepted-resume job contract drifted.")

    source_package = _mapping(
        job.get("source_package"), label="source package"
    )
    source_manifest_path = (
        root / SOURCE_PACKAGE_RELATIVE / "package_manifest.json"
    )
    if (
        source_package.get("path") != SOURCE_PACKAGE_RELATIVE.as_posix()
        or source_package.get("manifest_sha256")
        != SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
        or source_package.get("manifest_file_sha256")
        != SOURCE_PACKAGE_MANIFEST_FILE_SHA256
        or sha256_file(source_manifest_path)
        != SOURCE_PACKAGE_MANIFEST_FILE_SHA256
    ):
        raise ResumeContractError("Source package byte authority drifted.")
    source_manifest = load_json(
        source_manifest_path, label="source package manifest"
    )
    if (
        verify_self_digest(source_manifest, label="source package manifest")
        != SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
    ):
        raise ResumeContractError("Source package canonical authority drifted.")
    source_runner = root / SOURCE_PACKAGE_RELATIVE / "run_cell.py"
    source_archive = root / SOURCE_PACKAGE_RELATIVE / "source/source_locked.tar.gz"
    if (
        job.get("source_runner_sha256") != SOURCE_RUNNER_SHA256
        or sha256_file(source_runner) != SOURCE_RUNNER_SHA256
        or job.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or sha256_file(source_archive) != SOURCE_ARCHIVE_SHA256
    ):
        raise ResumeContractError("Source runtime authority drifted.")
    source_job_path, source_job = _verify_file_binding(
        job.get("source_job"), label="source job", canonical=True
    )
    source_protocol_path, source_protocol = _verify_file_binding(
        job.get("source_protocol"), label="source protocol", canonical=True
    )
    assert source_job is not None and source_protocol is not None
    if (
        source_job.get("execution_id") != job.get("source_execution_id")
        or source_job.get("protocol_sha256")
        != source_protocol.get("sha256")
        or source_job.get("protocol_file_sha256")
        != sha256_file(source_protocol_path)
        or source_protocol.get("sha256")
        != job.get("scientific_protocol_sha256")
        or source_protocol.get("horizon") != TARGET_HORIZON
        or source_protocol.get("request", {})
        .get("execution", {})
        .get("resume", {})
        .get("kind")
        != "fresh_start"
        or source_protocol.get("route_contract", {}).get("route_profile")
        != ROUTE_PROFILE
        or source_protocol.get("route_contract", {}).get("sha256")
        != ROUTE_CONTRACT_SHA256
    ):
        raise ResumeContractError("Source scientific protocol drifted.")
    archive_path, _ = _verify_file_binding(
        resume.get("archive"), label="resume archive"
    )
    if source_job_path != _runtime_path(
        job["source_job"]["path"], label="source job path"
    ) or archive_path != _runtime_path(
        resume["archive"]["path"], label="resume archive path"
    ):
        raise ResumeContractError("Runtime path resolution drifted.")
    return job, manifest


def _validate_authorization(
    path: Path, *, job: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = load_json(path, label="resume execution authorization")
    verify_self_digest(authorization, label="resume execution authorization")
    resume = _mapping(job.get("resume_input"), label="resume input")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != job.get("execution_id")
        or authorization.get("job_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != manifest.get("sha256")
        or authorization.get("scientific_protocol_sha256")
        != job.get("scientific_protocol_sha256")
        or authorization.get("checkpoint_sha256")
        != resume.get("checkpoint_sha256")
        or authorization.get("resources") != job.get("resources")
        or authorization.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authorization.get("scope")
        != "three_held_memory_repair_resumes_exact_cell_only"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("source_held_job_removal_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise ResumeContractError("Resume execution authorization drifted.")
    return authorization


def _load_source_runner() -> ModuleType:
    root = _runtime_root()
    source_dir = root / SOURCE_PACKAGE_RELATIVE
    path = source_dir / "run_cell.py"
    if sha256_file(path) != SOURCE_RUNNER_SHA256:
        raise ResumeContractError("Source runner bytes drifted before import.")
    source_text = str(source_dir)
    sys.path.insert(0, source_text)
    try:
        spec = importlib.util.spec_from_file_location(
            "paper_i_global_singleton_r50_source_runner_v2", path
        )
        if spec is None or spec.loader is None:
            raise ResumeContractError("Cannot load source runner.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == source_text:
            sys.path.pop(0)


def _prepare_source(job: Mapping[str, Any]) -> tuple[ModuleType, Any, Any, Any]:
    base = _load_source_runner()
    source_job_path = _runtime_path(
        job["source_job"]["path"], label="source job path"
    )
    source_job, _manifest, protocol, problem, temporary = base._prepare(
        source_job_path
    )
    if (
        source_job.get("execution_id") != job.get("source_execution_id")
        or source_job.get("sha256")
        != job.get("source_job", {}).get("canonical_sha256")
        or protocol.sha256 != job.get("scientific_protocol_sha256")
        or int(protocol.horizon) != TARGET_HORIZON
        or protocol.route_contract.get("route_profile") != ROUTE_PROFILE
        or protocol.route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
    ):
        temporary.cleanup()
        raise ResumeContractError("Prepared source protocol drifted.")
    return base, protocol, problem, temporary


def _extract_resume(job: Mapping[str, Any], destination: Path) -> Path:
    resume = _mapping(job.get("resume_input"), label="resume input")
    archive_path = _runtime_path(
        resume["archive"]["path"], label="resume archive path"
    )
    members_raw = _sequence(resume.get("members"), label="resume members")
    members = {
        safe_relative_path(row.get("path"), label="resume member path").as_posix(): row
        for row in members_raw
        if isinstance(row, Mapping)
    }
    if len(members) != len(members_raw) or len(members) != 3:
        raise ResumeContractError("Resume member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                expected = members.get(member.name)
                if (
                    expected is None
                    or member.name in observed
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != int(expected.get("size_bytes", -1))
                ):
                    raise ResumeContractError(
                        f"Unsafe or unexpected resume member: {member.name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ResumeContractError(
                        f"Unreadable resume member: {member.name}"
                    )
                target = destination / safe_relative_path(
                    member.name, label="resume member path"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = __import__("hashlib").sha256()
                size = 0
                with target.open("xb") as output:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    size != int(expected["size_bytes"])
                    or digest.hexdigest() != expected.get("sha256")
                ):
                    raise ResumeContractError(
                        f"Resume member digest drifted: {member.name}"
                    )
                observed.add(member.name)
        if observed != set(members):
            raise ResumeContractError("Resume archive member closure failed.")
        checkpoint = destination / safe_relative_path(
            resume.get("checkpoint_path"), label="resume checkpoint path"
        )
        if sha256_file(checkpoint) != resume.get("checkpoint_sha256"):
            raise ResumeContractError("Extracted checkpoint digest drifted.")
        return checkpoint
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        encoder = json.JSONEncoder(
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        for chunk in encoder.iterencode(dict(payload)):
            stream.write(chunk.encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def preflight(job_path: Path) -> dict[str, Any]:
    job, manifest = _load_runtime_job(job_path)
    _base, protocol, _problem, temporary = _prepare_source(job)
    try:
        return {
            "status": "passed",
            "package_id": PACKAGE_ID,
            "execution_id": job["execution_id"],
            "package_manifest_sha256": manifest["sha256"],
            "scientific_protocol_sha256": protocol.sha256,
            "scientific_protocol_changed": False,
            "scientific_settings_changed": [],
            "resume_controller_round": job["resume_input"][
                "resume_controller_round"
            ],
            "target_horizon": TARGET_HORIZON,
            "request_memory_mb": job["resources"]["request_memory_mb"],
            "request_disk_mb": job["resources"]["request_disk_mb"],
            "source_held_job_preserved": True,
        }
    finally:
        temporary.cleanup()


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    job, manifest = _load_runtime_job(job_path)
    authorization = _validate_authorization(
        authorization_path, job=job, manifest=manifest
    )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise ResumeContractError("Resume worker destination already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = output_dir.parent / f".{job['execution_id']}.resume_work_v1"
    if work_root.exists() or work_root.is_symlink():
        raise ResumeContractError("Resume worker staging already exists.")
    work_root.mkdir()
    staging = work_root / "artifacts"
    staging.mkdir()
    resume_root = work_root / "resume_input"
    checkpoint = _extract_resume(job, resume_root)
    base, protocol, problem, source_temporary = _prepare_source(job)
    try:
        from pipelines.static_adapt.ra_adapt import (
            RAAdaptOperationalControls,
            run_ra_adapt,
        )
        from pipelines.static_adapt.sr_snake import (
            AcceptedStateResume,
            CheckpointObservation,
            EstimatorLedgerObservation,
            SRObservationPolicy,
        )

        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=str(
                    job["resume_input"]["checkpoint_sha256"]
                ),
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=staging / "checkpoint.json",
                    every_controller_rounds=1,
                    keep_history_tail=100,
                ),
                estimator_ledger=EstimatorLedgerObservation(
                    path=staging / "estimator_ledger.json"
                ),
                resource_rounds=(TARGET_HORIZON,),
            ),
        )
        source_root = Path(source_temporary.name) / "source"
        base._reassert_source_root(source_root)
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result = run_ra_adapt(
                problem, protocol, operational_controls=controls
            )
        finally:
            os.chdir(original)
    finally:
        source_temporary.cleanup()

    rounds = len(result.run.accepted_trajectory)
    resume_round = int(job["resume_input"]["resume_controller_round"])
    if (
        result.protocol.sha256 != protocol.sha256
        or not resume_round < rounds <= TARGET_HORIZON
        or not (staging / "checkpoint.json").is_file()
        or not (staging / "estimator_ledger.json").is_file()
    ):
        raise ResumeContractError("Accepted resume result closure failed.")
    _write_json(staging / "result.json", result.to_dict())
    if result.run.paper_i_summary is None:
        raise ResumeContractError("Accepted resume summary is absent.")
    _write_json(
        staging / "paper_i_summary.json",
        result.run.paper_i_summary.to_dict(),
    )
    payloads = {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    execution_manifest = {
        "schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_"
            "resume128gb_execution_manifest_v1"
        ),
        "status": "passed",
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": job["execution_id"],
        "source_execution_id": job["source_execution_id"],
        "job_sha256": job["sha256"],
        "authorization_sha256": authorization["sha256"],
        "scientific_protocol_sha256": protocol.sha256,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
        "source_checkpoint_sha256": job["resume_input"][
            "checkpoint_sha256"
        ],
        "resume_controller_round": resume_round,
        "controller_rounds_completed": rounds,
        "target_horizon": TARGET_HORIZON,
        "source_held_job_preserved": True,
        "output_payloads": payloads,
    }
    execution_manifest["sha256"] = __import__("hashlib").sha256(
        canonical_json_bytes(execution_manifest)
    ).hexdigest()
    _write_json(staging / "execution_manifest.json", execution_manifest)
    os.rename(staging, output_dir)
    shutil.rmtree(resume_root, ignore_errors=True)
    try:
        work_root.rmdir()
    except OSError:
        pass
    receipt = {
        "schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_"
            "resume128gb_worker_receipt_v1"
        ),
        "status": "passed",
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": job["execution_id"],
        "job_sha256": job["sha256"],
        "authorization_sha256": authorization["sha256"],
        "execution_manifest_sha256": execution_manifest["sha256"],
        "controller_rounds_completed": rounds,
        "source_checkpoint_consumed": True,
        "source_checkpoint_sha256": job["resume_input"][
            "checkpoint_sha256"
        ],
        "source_held_job_preserved": True,
        "artifacts": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        ],
    }
    receipt["sha256"] = __import__("hashlib").sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.preflight:
            if any(
                value is not None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                )
            ):
                raise ResumeContractError(
                    "Preflight accepts no execution destinations."
                )
            payload = preflight(args.job.resolve())
        else:
            if any(
                value is None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                )
            ):
                raise ResumeContractError(
                    "Execution requires authorization and destinations."
                )
            payload = run_cell(
                job_path=args.job.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
    except (OSError, ResumeContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
