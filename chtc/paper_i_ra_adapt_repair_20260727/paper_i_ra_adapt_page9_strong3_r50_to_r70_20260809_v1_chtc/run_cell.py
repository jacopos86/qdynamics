#!/usr/bin/env python3
"""Preflight or execute one authenticated Page-9 k=50 -> 70 continuation."""

from __future__ import annotations

import argparse
import errno
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
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
    ALGORITHM_ID,
    AUTHORIZATION_SCHEMA,
    BASE_PACKAGE_MANIFEST_CANONICAL_SHA256,
    BASE_PACKAGE_MANIFEST_FILE_SHA256,
    BASE_PACKAGE_RELATIVE,
    BASE_RUNNER_SHA256,
    BASE_SOURCE_ARCHIVE_SHA256,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_BEFORE_SHA256,
    CONTROLLER_RELATIVE_PATH,
    CONTROLLER_REPAIR_ID,
    CONTROL_FILES,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_SCHEMA,
    REMOTE_IMAGE_SHA256,
    RESOURCE_ENVELOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_PROFILE,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    prefix_projection,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_resume_archive,
    verify_self_digest,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _package_path(value: Any, *, label: str) -> Path:
    path = PACKAGE_DIR / safe_relative_path(value, label=label)
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped package.") from exc
    return path


def _verify_binding(
    raw: Any, *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = _package_path(row.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"{label} binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != row.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_job(job_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("status")
        != "passed_inert_blocked_1_of_3_resume_inputs"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("row_count") != 3
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("blocked_execution_ids")
        != [expected_execution_ids()[2]]
        or manifest.get("source_horizon") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("route_profile") != ROUTE_PROFILE
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert package identity drifted.")
    controls = _sequence(manifest.get("control_files"), label="control files")
    if len(controls) != len(CONTROL_FILES):
        raise PackageContractError("Control-plane cardinality drifted.")
    for row in controls:
        _verify_binding(row, label="control file")
    matches = []
    for raw in _sequence(manifest.get("jobs"), label="job bindings"):
        if not isinstance(raw, Mapping):
            continue
        path = _package_path(raw.get("path"), label="job path")
        if path.resolve() == job_path.resolve():
            matches.append(raw)
    if len(matches) != 1:
        raise PackageContractError("Requested job is outside package closure.")
    _bound, payload = _verify_binding(matches[0], label="job", canonical=True)
    assert payload is not None
    job = payload
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") not in expected_execution_ids()
        or job.get("execution_mode") != "authenticated_accepted_state_resume"
        or job.get("source_horizon") != SOURCE_HORIZON
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("only_scientific_change")
        != "maximum_controller_rounds_50_to_70"
        or job.get("route_profile") != ROUTE_PROFILE
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("algorithm_id") != ALGORITHM_ID
        or job.get("candidate_adapter_id") != CANDIDATE_ADAPTER_ID
        or job.get("resources") != RESOURCE_ENVELOPE
        or job.get("accepted_state_resume_required") is not True
        or job.get("triplet_pointer_closure_required") is not True
        or job.get("prefix_equality_required") is not True
        or job.get("accepted_energy_roundoff_overlay", {}).get("repair_id")
        != CONTROLLER_REPAIR_ID
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Continuation job identity drifted.")
    protocol_path, protocol = _verify_binding(
        job.get("derived_protocol"), label="derived protocol", canonical=True
    )
    assert protocol is not None
    if (
        protocol_path != _package_path(
            job["derived_protocol"]["path"], label="derived protocol path"
        )
        or protocol.get("sha256") != job.get("derived_protocol_sha256")
    ):
        raise PackageContractError("Job-to-protocol binding drifted.")
    return job, manifest, protocol


def _base_runner(base_package_dir: Path) -> ModuleType:
    manifest_path = base_package_dir / "package_manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or sha256_file(manifest_path) != BASE_PACKAGE_MANIFEST_FILE_SHA256
    ):
        raise PackageContractError("Page-9 base package bytes drifted.")
    manifest = load_json(manifest_path, label="Page-9 base package manifest")
    if (
        verify_self_digest(manifest, label="Page-9 base package manifest")
        != BASE_PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("source_archive", {}).get("sha256")
        != BASE_SOURCE_ARCHIVE_SHA256
    ):
        raise PackageContractError("Page-9 base package identity drifted.")
    runner_path = base_package_dir / "run_cell.py"
    if sha256_file(runner_path) != BASE_RUNNER_SHA256:
        raise PackageContractError("Page-9 base runner drifted.")
    local_contract = sys.modules.get("package_contract")
    original_path = list(sys.path)
    sys.modules.pop("package_contract", None)
    sys.path.insert(0, base_package_dir.as_posix())
    try:
        spec = importlib.util.spec_from_file_location(
            "page9_v3_runtime_runner", runner_path
        )
        if spec is None or spec.loader is None:
            raise PackageContractError("Cannot import Page-9 base runner.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop("package_contract", None)
        if local_contract is not None:
            sys.modules["package_contract"] = local_contract


def _apply_controller_overlay(source_root: Path, manifest: Mapping[str, Any]) -> None:
    _composition_path, composition = _verify_binding(
        manifest.get("source_composition"), label="source composition", canonical=True
    )
    assert composition is not None
    overlay = _mapping(
        composition.get("operational_overlay"), label="operational overlay"
    )
    after = _mapping(overlay.get("after"), label="overlay file")
    overlay_path = _package_path(after.get("path"), label="overlay file path")
    controller = source_root / CONTROLLER_RELATIVE_PATH
    if (
        composition.get("base_source_archive_sha256")
        != BASE_SOURCE_ARCHIVE_SHA256
        or overlay.get("repair_id") != CONTROLLER_REPAIR_ID
        or overlay.get("path") != CONTROLLER_RELATIVE_PATH
        or overlay.get("before_sha256") != CONTROLLER_BEFORE_SHA256
        or overlay.get("all_non_energy_fields_exact") is not True
        or sha256_file(controller) != CONTROLLER_BEFORE_SHA256
        or sha256_file(overlay_path) != CONTROLLER_AFTER_SHA256
    ):
        raise PackageContractError("Accepted-energy-only overlay drifted.")
    temporary = controller.with_name(f".{controller.name}.overlay")
    shutil.copyfile(overlay_path, temporary)
    os.replace(temporary, controller)
    if sha256_file(controller) != CONTROLLER_AFTER_SHA256:
        raise PackageContractError("Accepted-energy-only overlay failed.")


def _source_job_path(base_package_dir: Path, job: Mapping[str, Any]) -> Path:
    relative = safe_relative_path(job["source_job"]["path"], label="source job path")
    expected_prefix = BASE_PACKAGE_RELATIVE.parts
    if relative.parts[: len(expected_prefix)] != expected_prefix:
        raise PackageContractError("Source job escaped the Page-9 package.")
    path = base_package_dir.joinpath(*relative.parts[len(expected_prefix) :])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(job["source_job"]["size_bytes"])
        or sha256_file(path) != job["source_job"]["sha256"]
    ):
        raise PackageContractError("Source job bytes drifted.")
    return path


def _prepare(
    *,
    job_path: Path,
    base_package_dir: Path,
    temporary_parent: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, tempfile.TemporaryDirectory[str]]:
    job, manifest, derived_payload = _load_job(job_path)
    base = _base_runner(base_package_dir)
    source_job_path = _source_job_path(base_package_dir, job)
    base_job, base_manifest, source_payload, locks = base._load_closed_job(
        source_job_path
    )
    temporary = tempfile.TemporaryDirectory(
        prefix=f".paper-i-page9-r70-{job['regime_id']}.",
        dir=temporary_parent,
    )
    try:
        source_root = Path(temporary.name) / "source"
        base._extract_source(
            manifest=base_manifest,
            source_locks=locks,
            destination=source_root,
        )
        _apply_controller_overlay(source_root, manifest)
        original = Path.cwd()
        os.chdir(source_root)
        try:
            base._activate_source_root(source_root)
            source, problem = base._load_protocol(
                job=base_job,
                payload=source_payload,
                source_locks=locks,
            )
            from pipelines.static_adapt.ra_adapt.contracts import (
                _attach_validated_bundle_protocol_authority,
                _mint_bundle_protocol_materialization_authority,
                resolved_ra_adapt_protocol_from_mapping,
            )

            derived = resolved_ra_adapt_protocol_from_mapping(derived_payload)
            receipt = derived.bundle_materialization
            if receipt is None:
                raise PackageContractError("Derived protocol lost bundle authority.")
            authority = _mint_bundle_protocol_materialization_authority(
                receipt,
                source_lock_refs=derived.source_locks,
                protocol_sha256=derived.sha256,
            )
            derived = _attach_validated_bundle_protocol_authority(
                derived, authority
            )
        finally:
            os.chdir(original)
        source_request = source.request.to_dict()
        derived_request = derived.request.to_dict()
        source_request["execution"]["stop"].pop(
            "maximum_controller_rounds", None
        )
        derived_request["execution"]["stop"].pop(
            "maximum_controller_rounds", None
        )
        if (
            source.sha256 != job.get("source_protocol_sha256")
            or derived.sha256 != job.get("derived_protocol_sha256")
            or source.route_contract != derived.route_contract
            or derived.route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
            or derived.route_contract.get("route_profile") != ROUTE_PROFILE
            or source_request != derived_request
            or source.source_locks != derived.source_locks
            or int(source.horizon) != SOURCE_HORIZON
            or int(derived.horizon) != TARGET_HORIZON
            or derived.request.execution.stop.maximum_controller_rounds
            != TARGET_HORIZON
            or receipt.bundle_id != BUNDLE_ID
            or receipt.cell_id != job.get("execution_id")
            or derived.algorithm_id != ALGORITHM_ID
            or derived.adapter_id != CANDIDATE_ADAPTER_ID
        ):
            raise PackageContractError("Source-to-continuation protocol drifted.")
    except BaseException:
        temporary.cleanup()
        raise
    return job, manifest, derived, problem, temporary


def _validate_materialization(
    *,
    job: Mapping[str, Any],
    manifest_path: Path,
    archive_path: Path,
) -> dict[str, Any]:
    materialization = load_json(manifest_path, label="resume materialization")
    validate_resume_archive(
        archive_path, materialization, expected_round=SOURCE_HORIZON
    )
    anchor = materialization.get("prefix_anchor")
    if not isinstance(anchor, Mapping):
        raise PackageContractError("Resume prefix anchor is absent.")
    verify_self_digest(anchor, label="resume prefix anchor")
    source = _mapping(job.get("resume_source"), label="job resume source")
    if source.get("state") == "remote_archive_preserved_materialization_pending":
        bound = _mapping(source.get("prefix_anchor"), label="job prefix anchor")
        path = _package_path(bound.get("path"), label="job prefix anchor path")
        if (
            sha256_file(path) != bound.get("sha256")
            or load_json(path, label="job prefix anchor").get("sha256")
            != anchor.get("sha256")
            or materialization.get("source_archive")
            != source.get("remote_full_archive")
        ):
            raise PackageContractError("Materialized source/prefix binding drifted.")
    elif source.get("state") == "blocked_predecessor_terminal_missing":
        completion = materialization.get("source_completion_bindings")
        if not isinstance(completion, Mapping) or not completion:
            raise PackageContractError(
                "Blocked predecessor lacks completion authority."
            )
    else:
        raise PackageContractError("Unknown job resume-source state.")
    if (
        materialization.get("package_id") != PACKAGE_ID
        or materialization.get("execution_id") != job.get("execution_id")
        or materialization.get("source_execution_id")
        != job.get("source_execution_id")
        or materialization.get("source_job_sha256")
        != job.get("source_job", {}).get("canonical_sha256")
    ):
        raise PackageContractError("Resume materialization identity drifted.")
    return materialization


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    authority = load_json(path, label="execution authorization")
    verify_self_digest(authority, label="execution authorization")
    if (
        authority.get("schema") != AUTHORIZATION_SCHEMA
        or authority.get("package_id") != PACKAGE_ID
        or authority.get("campaign_id") != CAMPAIGN_ID
        or authority.get("execution_id") != job.get("execution_id")
        or authority.get("job_sha256") != job.get("sha256")
        or authority.get("package_manifest_sha256") != manifest.get("sha256")
        or authority.get("derived_protocol_sha256")
        != job.get("derived_protocol_sha256")
        or authority.get("resume_materialization_sha256")
        != materialization.get("sha256")
        or authority.get("resume_archive_sha256")
        != materialization.get("archive", {}).get("sha256")
        or authority.get("pinned_image_sha256") != REMOTE_IMAGE_SHA256
        or authority.get("resources") != RESOURCE_ENVELOPE
        or authority.get("scope") != "single_page9_strong_sector_r70_cell"
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not True
        or authority.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError("Execution authorization drifted.")
    return authority


def _extract_resume(
    *,
    archive_path: Path,
    materialization: Mapping[str, Any],
    destination: Path,
) -> Path:
    validated = validate_resume_archive(
        archive_path, materialization, expected_round=SOURCE_HORIZON
    )
    rows = {
        str(row["path"]): row
        for row in materialization["members"]
        if isinstance(row, Mapping)
    }
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label="resume tar member"
            ).as_posix()
            row = rows.get(relative)
            if (
                row is None
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row["size_bytes"])
            ):
                raise PackageContractError(f"Unsafe resume member: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(f"Unreadable resume member: {relative}")
            target = destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = __import__("hashlib").sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row["sha256"]:
                raise PackageContractError(f"Extracted resume drifted: {relative}")
            observed.add(relative)
    if observed != set(rows):
        raise PackageContractError("Resume extraction is incomplete.")
    checkpoint = destination / str(
        validated["members_by_role"]["checkpoint"]["path"]
    )
    return checkpoint


def preflight(
    *,
    job_path: Path,
    base_package_dir: Path,
    resume_manifest: Path | None,
    resume_archive: Path | None,
) -> dict[str, Any]:
    job, manifest, protocol, _problem, temporary = _prepare(
        job_path=job_path,
        base_package_dir=base_package_dir,
    )
    try:
        materialized = False
        if resume_manifest is not None or resume_archive is not None:
            if resume_manifest is None or resume_archive is None:
                raise PackageContractError(
                    "Resume preflight requires both manifest and archive."
                )
            _validate_materialization(
                job=job,
                manifest_path=resume_manifest,
                archive_path=resume_archive,
            )
            materialized = True
        return digested(
            {
                "schema": "paper_i_page9_strong3_r70_worker_preflight_v1",
                "status": "passed" if materialized else "blocked_resume_input_pending",
                "package_id": PACKAGE_ID,
                "execution_id": job["execution_id"],
                "package_manifest_sha256": manifest["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "resume_source_state": job["resume_source"]["state"],
                "resume_materialized": materialized,
                "accepted_energy_roundoff_overlay": CONTROLLER_REPAIR_ID,
                "target_horizon": TARGET_HORIZON,
                "scientific_execution_performed": False,
            }
        )
    finally:
        temporary.cleanup()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_publish_tree(source: Path, destination: Path) -> None:
    """Publish a result tree atomically, with an EXDEV-safe copy fallback."""

    if destination.exists() or destination.is_symlink():
        raise PackageContractError(f"Refusing existing result tree: {destination}")
    temporary = destination.with_name(f".{destination.name}.publish-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PackageContractError(f"Refusing stale publish tree: {temporary}")
    try:
        try:
            os.rename(source, destination)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        shutil.copytree(source, temporary)
        os.rename(temporary, destination)
        shutil.rmtree(source)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run_cell(
    *,
    job_path: Path,
    base_package_dir: Path,
    resume_manifest_path: Path,
    resume_archive_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    job, manifest, protocol, problem, temporary = _prepare(
        job_path=job_path,
        base_package_dir=base_package_dir,
        temporary_parent=output_dir.parent,
    )
    try:
        materialization = _validate_materialization(
            job=job,
            manifest_path=resume_manifest_path,
            archive_path=resume_archive_path,
        )
        authority = _validate_authorization(
            authorization_path,
            job=job,
            manifest=manifest,
            materialization=materialization,
        )
        if (
            output_dir.exists()
            or output_dir.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
            or output_dir.name != job["execution_id"]
            or output_dir.parent.name != "runs"
        ):
            raise PackageContractError("Worker output destination is unsafe.")
        root = Path(temporary.name)
        checkpoint = _extract_resume(
            archive_path=resume_archive_path,
            materialization=materialization,
            destination=root / "resume_input",
        )
        staging = root / "cell_output"
        (staging / "checkpoints").mkdir(parents=True)
        (staging / "result").mkdir(parents=True)
        (staging / "summary").mkdir(parents=True)
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

        checkpoint_row = next(
            row
            for row in materialization["members"]
            if row["role"] == "checkpoint"
        )
        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=str(checkpoint_row["sha256"]),
            ),
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=staging / "checkpoints/current.json",
                    every_controller_rounds=1,
                    keep_history_tail=100,
                ),
                estimator_ledger=EstimatorLedgerObservation(
                    path=staging / "result/estimator_ledger.json"
                ),
                resource_rounds=(SOURCE_HORIZON, TARGET_HORIZON),
            ),
        )
        source_root = root / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result = run_ra_adapt(
                problem, protocol, operational_controls=controls
            )
        finally:
            os.chdir(original)
        rounds = len(result.run.accepted_trajectory)
        if (
            result.protocol.sha256 != protocol.sha256
            or rounds != TARGET_HORIZON
            or not (staging / "checkpoints/current.json").is_file()
            or not (staging / "result/estimator_ledger.json").is_file()
            or result.run.paper_i_summary is None
        ):
            raise PackageContractError(
                f"Continuation stopped at round {rounds}, not {TARGET_HORIZON}."
            )
        summary = result.run.paper_i_summary.to_dict()
        observed_prefix = prefix_projection(summary)
        if observed_prefix.get("sha256") != materialization["prefix_anchor"].get(
            "sha256"
        ):
            raise PackageContractError("Resumed accepted prefix is not exactly equal.")
        _write_json(staging / "result/result.json", result.to_dict())
        _write_json(staging / "summary/summary.json", summary)
        outputs = {
            "checkpoint": staging / "checkpoints/current.json",
            "estimator_ledger": staging / "result/estimator_ledger.json",
            "result": staging / "result/result.json",
            "summary": staging / "summary/summary.json",
        }
        execution_manifest = digested(
            {
                "schema": "paper_i_page9_strong3_r70_execution_manifest_v1",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "resume_materialization_sha256": materialization["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "controller_rounds_completed": rounds,
                "accepted_state_resume": True,
                "triplet_pointer_closed": True,
                "prefix_equality_passed": True,
                "accepted_energy_roundoff_overlay": CONTROLLER_REPAIR_ID,
                "output_payloads": {
                    role: {
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for role, path in outputs.items()
                },
            }
        )
        _write_json(staging / "execution_manifest.json", execution_manifest)
        _atomic_publish_tree(staging, output_dir)
        receipt = digested(
            {
                "schema": "paper_i_page9_strong3_r70_worker_receipt_v1",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "accepted_state_resume": True,
                "prefix_equality_passed": True,
                "artifacts": [
                    {
                        "path": (
                            PurePosixPath("runs")
                            / job["execution_id"]
                            / path.relative_to(output_dir)
                        ).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(output_dir.rglob("*"))
                    if path.is_file()
                ],
            }
        )
        _write_json(receipt_path, receipt)
        return receipt
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--base-package-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--resume-materialization", type=Path)
    parser.add_argument("--resume-archive", type=Path)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        base = (
            args.base_package_dir.resolve()
            if args.base_package_dir is not None
            else (
                repo_root_from_script(__file__) / BASE_PACKAGE_RELATIVE
            ).resolve()
        )
        if args.preflight:
            if any(
                value is not None
                for value in (
                    args.execution_authorization,
                    args.output_dir,
                    args.receipt,
                )
            ):
                raise PackageContractError("Preflight accepts no output authority.")
            result = preflight(
                job_path=args.job.resolve(),
                base_package_dir=base,
                resume_manifest=(
                    None
                    if args.resume_materialization is None
                    else args.resume_materialization.resolve()
                ),
                resume_archive=(
                    None
                    if args.resume_archive is None
                    else args.resume_archive.resolve()
                ),
            )
        else:
            required = (
                args.resume_materialization,
                args.resume_archive,
                args.execution_authorization,
                args.output_dir,
                args.receipt,
            )
            if any(value is None for value in required):
                raise PackageContractError(
                    "Execution requires resume, authorization, and destinations."
                )
            assert args.resume_materialization is not None
            assert args.resume_archive is not None
            assert args.execution_authorization is not None
            assert args.output_dir is not None
            assert args.receipt is not None
            result = run_cell(
                job_path=args.job.resolve(),
                base_package_dir=base,
                resume_manifest_path=args.resume_materialization.resolve(),
                resume_archive_path=args.resume_archive.resolve(),
                authorization_path=args.execution_authorization.resolve(),
                output_dir=args.output_dir.resolve(),
                receipt_path=args.receipt.resolve(),
            )
    except (OSError, ValueError, PackageContractError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
