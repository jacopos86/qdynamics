#!/usr/bin/env python3
"""Execute one sealed exact-prefix global-singleton r70 continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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

from package_contract import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    CONTROL_FILES,
    HORIZON_CHANGED_PATHS,
    JOB_SCHEMA,
    LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256,
    LOADER_PACKAGE_MANIFEST_FILE_SHA256,
    LOADER_PACKAGE_RELATIVE,
    LOADER_RUNNER_SHA256,
    PACKAGE_ID,
    PACKAGE_RELATIVE,
    PACKAGE_SCHEMA,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    SCIENTIFIC_SETTINGS_CHANGED,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    implementation_repair,
    load_json,
    safe_relative_path,
    scalar_differences,
    sha256_file,
    verify_self_digest,
)


def _runtime_root() -> Path:
    root = PACKAGE_DIR
    for _part in PACKAGE_RELATIVE.parts:
        root = root.parent
    if root / PACKAGE_RELATIVE != PACKAGE_DIR:
        raise PackageContractError("Runtime package path is not repo-relative.")
    return root


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _runtime_path(value: Any, *, label: str) -> Path:
    root = _runtime_root()
    path = root / safe_relative_path(value, label=label)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped runtime root.") from exc
    return path


def _verify_json_binding(
    raw: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    binding = _mapping(raw, label=f"{label} binding")
    path = _runtime_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} exact bytes drifted.")
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_job(job_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    manifest = load_json(manifest_path, label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "passed_inert_three_authenticated_r70_resumes"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("row_count") != 3
        or manifest.get("source_horizon") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or manifest.get("changed_protocol_paths")
        != list(HORIZON_CHANGED_PATHS)
        or manifest.get("non_swept_settings_diff") != []
        or manifest.get("implementation_repair") != implementation_repair()
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package manifest identity drifted.")
    controls = _sequence(manifest.get("control_files"), label="controls")
    if len(controls) != len(CONTROL_FILES):
        raise PackageContractError("Control-plane cardinality drifted.")
    for binding in controls:
        row = _mapping(binding, label="control binding")
        relative = safe_relative_path(row.get("path"), label="control path")
        if len(relative.parts) != 1:
            raise PackageContractError("Control path drifted.")
        path = PACKAGE_DIR / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise PackageContractError("Control-plane bytes drifted.")
    resolved = job_path.resolve()
    candidates: list[Mapping[str, Any]] = []
    for raw in _sequence(manifest.get("jobs"), label="job bindings"):
        if not isinstance(raw, Mapping):
            continue
        candidate = PACKAGE_DIR / safe_relative_path(
            raw.get("path"), label="job path"
        )
        if candidate.resolve() == resolved:
            candidates.append(raw)
    if len(candidates) != 1:
        raise PackageContractError("Job is outside the package closure.")
    binding = candidates[0]
    if (
        not job_path.is_file()
        or job_path.is_symlink()
        or job_path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(job_path) != binding.get("sha256")
    ):
        raise PackageContractError("Job byte binding drifted.")
    job = load_json(job_path, label="job")
    if verify_self_digest(job, label="job") != binding.get("canonical_sha256"):
        raise PackageContractError("Job digest drifted.")
    resume = _mapping(job.get("resume_input"), label="resume input")
    members = _sequence(resume.get("members"), label="resume members")
    roles = {row.get("role") for row in members if isinstance(row, Mapping)}
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_mode")
        != "authenticated_exact_prefix_resume_to_70"
        or job.get("source_horizon") != SOURCE_HORIZON
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("route_profile") != ROUTE_PROFILE
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("resources") != RESOURCE_ENVELOPE
        or job.get("implementation_repair") != implementation_repair()
        or job.get("scientific_protocol_changed") is not True
        or job.get("scientific_settings_changed")
        != list(SCIENTIFIC_SETTINGS_CHANGED)
        or job.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or job.get("changed_protocol_paths")
        != list(HORIZON_CHANGED_PATHS)
        or job.get("non_swept_settings_diff") != []
        or job.get("source_held_job_preserved") is not True
        or resume.get("pointer_closed") is not True
        or resume.get("member_count") != 3
        or len(members) != 3
        or roles
        != {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        or not 0
        < int(resume.get("resume_controller_round", -1))
        < TARGET_HORIZON
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Job contract drifted.")
    return job, manifest


def _validate_authorization(
    path: Path, *, job: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = load_json(path, label="execution authorization")
    verify_self_digest(authorization, label="execution authorization")
    resume = _mapping(job.get("resume_input"), label="resume input")
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != job.get("execution_id")
        or authorization.get("job_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != manifest.get("sha256")
        or authorization.get("derived_protocol_sha256")
        != job.get("derived_protocol_sha256")
        or authorization.get("checkpoint_sha256")
        != resume.get("checkpoint_sha256")
        or authorization.get("resume_controller_round")
        != resume.get("resume_controller_round")
        or authorization.get("target_horizon") != TARGET_HORIZON
        or authorization.get("resources") != RESOURCE_ENVELOPE
        or authorization.get("scope") != "weak_strong_exact_prefix_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("source_held_job_removal_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError("Execution authorization drifted.")
    return authorization


def _loader_module() -> ModuleType:
    root = _runtime_root()
    manifest_path = root / LOADER_PACKAGE_RELATIVE / "package_manifest.json"
    if sha256_file(manifest_path) != LOADER_PACKAGE_MANIFEST_FILE_SHA256:
        raise PackageContractError("Loader package manifest bytes drifted.")
    manifest = load_json(manifest_path, label="loader package manifest")
    if (
        verify_self_digest(manifest, label="loader package manifest")
        != LOADER_PACKAGE_MANIFEST_CANONICAL_SHA256
    ):
        raise PackageContractError("Loader package manifest digest drifted.")
    path = root / LOADER_PACKAGE_RELATIVE / "run_resume_cell.py"
    if sha256_file(path) != LOADER_RUNNER_SHA256:
        raise PackageContractError("Loader runner bytes drifted.")
    spec = importlib.util.spec_from_file_location(
        "paper_i_global_singleton_loaderfix_v2_runtime", path
    )
    if spec is None or spec.loader is None:
        raise PackageContractError("Cannot import loader runtime.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_source(job: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    _predecessor_path, predecessor = _verify_json_binding(
        job.get("predecessor_job"), label="predecessor job"
    )
    if (
        predecessor.get("execution_id")
        != job.get("predecessor_execution_id")
        or predecessor.get("source_execution_id")
        != job.get("base_execution_id")
        or predecessor.get("implementation_repair")
        != implementation_repair()
    ):
        raise PackageContractError("Predecessor job identity drifted.")
    loader = _loader_module()
    local_contract = sys.modules.pop("package_contract", None)
    try:
        base, source, problem, temporary = loader._prepare_source(predecessor)
    finally:
        sys.modules.pop("package_contract", None)
        if local_contract is not None:
            sys.modules["package_contract"] = local_contract
    try:
        _derived_path, derived_payload = _verify_json_binding(
            job.get("derived_protocol"), label="derived protocol"
        )
        from pipelines.static_adapt.ra_adapt.contracts import (
            _attach_validated_bundle_protocol_authority,
            _mint_bundle_protocol_materialization_authority,
            resolved_ra_adapt_protocol_from_mapping,
        )

        derived = resolved_ra_adapt_protocol_from_mapping(derived_payload)
        if derived.bundle_materialization is None:
            raise PackageContractError("Derived protocol lost materialization.")
        authority = _mint_bundle_protocol_materialization_authority(
            derived.bundle_materialization,
            source_lock_refs=derived.source_locks,
            protocol_sha256=derived.sha256,
        )
        derived = _attach_validated_bundle_protocol_authority(
            derived, authority
        )
        changed = sorted(
            ".".join(str(component) for component in path)
            for path, _before, _after in scalar_differences(
                source.to_dict(), derived.to_dict()
            )
        )
        if (
            source.sha256 != job.get("source_protocol_sha256")
            or derived.sha256 != job.get("derived_protocol_sha256")
            or changed != sorted(HORIZON_CHANGED_PATHS)
            or int(source.horizon) != SOURCE_HORIZON
            or int(derived.horizon) != TARGET_HORIZON
            or source.route_contract != derived.route_contract
            or derived.route_contract.get("route_profile") != ROUTE_PROFILE
            or derived.route_contract.get("sha256")
            != ROUTE_CONTRACT_SHA256
            or source.bundle_materialization != derived.bundle_materialization
            or source.source_locks != derived.source_locks
        ):
            raise PackageContractError(
                f"Source-to-r70 protocol closure drifted: {changed}"
            )
    except BaseException:
        temporary.cleanup()
        raise
    return base, source, derived, problem, temporary


def _extract_resume(job: Mapping[str, Any], destination: Path) -> Path:
    resume = _mapping(job.get("resume_input"), label="resume input")
    archive = _runtime_root() / safe_relative_path(
        resume.get("runtime_archive_basename"),
        label="runtime resume archive",
    )
    bound = _mapping(resume.get("local_archive"), label="archive binding")
    if (
        not archive.is_file()
        or archive.is_symlink()
        or archive.stat().st_size != int(bound.get("size_bytes", -1))
        or sha256_file(archive) != bound.get("sha256")
    ):
        raise PackageContractError("Runtime resume archive bytes drifted.")
    raw_members = _sequence(resume.get("members"), label="resume members")
    members = {
        safe_relative_path(row.get("path"), label="resume member").as_posix(): row
        for row in raw_members
        if isinstance(row, Mapping)
    }
    if len(members) != 3 or len(members) != len(raw_members):
        raise PackageContractError("Resume member index drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    try:
        with tarfile.open(archive, mode="r|gz") as stream:
            for member in stream:
                binding = members.get(member.name)
                if (
                    binding is None
                    or member.name in observed
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != int(binding.get("size_bytes", -1))
                ):
                    raise PackageContractError(
                        f"Unsafe resume member: {member.name}"
                    )
                source = stream.extractfile(member)
                if source is None:
                    raise PackageContractError(
                        f"Unreadable resume member: {member.name}"
                    )
                target = destination / safe_relative_path(
                    member.name, label="resume member"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                    output.flush()
                    os.fsync(output.fileno())
                if (
                    size != member.size
                    or digest.hexdigest() != binding.get("sha256")
                ):
                    raise PackageContractError(
                        f"Resume member digest drifted: {member.name}"
                    )
                observed.add(member.name)
        if observed != set(members):
            raise PackageContractError("Resume archive closure is incomplete.")
        checkpoint = destination / safe_relative_path(
            resume.get("checkpoint_path"), label="checkpoint path"
        )
        if sha256_file(checkpoint) != resume.get("checkpoint_sha256"):
            raise PackageContractError("Checkpoint digest drifted.")
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
            stream.write(chunk.encode("ascii"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def preflight(job_path: Path) -> dict[str, Any]:
    job, manifest = _load_job(job_path)
    _base, source, derived, _problem, temporary = _prepare_source(job)
    try:
        return {
            "status": "passed",
            "package_id": PACKAGE_ID,
            "execution_id": job["execution_id"],
            "package_manifest_sha256": manifest["sha256"],
            "source_protocol_sha256": source.sha256,
            "derived_protocol_sha256": derived.sha256,
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
            "non_swept_settings_diff": [],
            "resume_controller_round": job["resume_input"][
                "resume_controller_round"
            ],
            "target_horizon": TARGET_HORIZON,
            "request_memory_mb": RESOURCE_ENVELOPE["request_memory_mb"],
            "request_disk_mb": RESOURCE_ENVELOPE["request_disk_mb"],
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
    job, manifest = _load_job(job_path)
    authorization = _validate_authorization(
        authorization_path, job=job, manifest=manifest
    )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PackageContractError("Worker destination already exists.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = output_dir.parent / f".{job['execution_id']}.resume_work_v1"
    if work_root.exists() or work_root.is_symlink():
        raise PackageContractError("Worker staging already exists.")
    work_root.mkdir()
    staging = work_root / "artifacts"
    staging.mkdir()
    resume_root = work_root / "resume_input"
    checkpoint = _extract_resume(job, resume_root)
    base, source, derived, problem, source_temporary = _prepare_source(job)
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
                checkpoint_sha256=job["resume_input"]["checkpoint_sha256"],
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
                problem, derived, operational_controls=controls
            )
        finally:
            os.chdir(original)
    finally:
        source_temporary.cleanup()
    rounds = len(result.run.accepted_trajectory)
    resume_round = int(job["resume_input"]["resume_controller_round"])
    if (
        result.protocol.sha256 != derived.sha256
        or not resume_round <= rounds <= TARGET_HORIZON
        or not (staging / "checkpoint.json").is_file()
        or not (staging / "estimator_ledger.json").is_file()
    ):
        raise PackageContractError("Continuation result closure failed.")
    _write_json(staging / "result.json", result.to_dict())
    if result.run.paper_i_summary is None:
        raise PackageContractError("Continuation summary is absent.")
    _write_json(
        staging / "paper_i_summary.json",
        result.run.paper_i_summary.to_dict(),
    )
    execution_manifest = {
        "schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_"
            "r70_execution_manifest_v1"
        ),
        "status": "passed",
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "execution_id": job["execution_id"],
        "job_sha256": job["sha256"],
        "authorization_sha256": authorization["sha256"],
        "source_protocol_sha256": source.sha256,
        "derived_protocol_sha256": derived.sha256,
        "source_checkpoint_sha256": job["resume_input"][
            "checkpoint_sha256"
        ],
        "resume_controller_round": resume_round,
        "controller_rounds_completed": rounds,
        "source_horizon": SOURCE_HORIZON,
        "target_horizon": TARGET_HORIZON,
        "only_scientific_change": "maximum_controller_rounds_50_to_70",
        "changed_protocol_paths": list(HORIZON_CHANGED_PATHS),
        "non_swept_settings_diff": [],
        "source_held_job_preserved": True,
        "output_payloads": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        },
    }
    execution_manifest["sha256"] = hashlib.sha256(
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
            "r70_worker_receipt_v1"
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
    receipt["sha256"] = hashlib.sha256(
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
                raise PackageContractError(
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
                raise PackageContractError(
                    "Execution requires authorization and destinations."
                )
            payload = run_cell(
                job_path=args.job.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
    except (OSError, ValueError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
