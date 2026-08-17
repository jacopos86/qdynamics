#!/usr/bin/env python3
"""Execute one authenticated Page-12 accepted-state continuation to round 70."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ALGORITHM_ID,
    AUTHORIZATION_SCHEMA,
    BACKEND_COMPILE_SCOPE,
    BASE_SOURCE_ARCHIVE_SHA256,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_BEFORE_SHA256,
    CONTROLLER_RELATIVE_PATH,
    CONTROLLER_REPAIR_ID,
    CONTINUATION_ROW_COUNT,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    RESOURCE_ENVELOPE,
    RESOURCE_WEIGHTING_SCOPE,
    REMOTE_IMAGE_SHA256,
    RESUME_AFTER_SHA256,
    RESUME_BEFORE_SHA256,
    RESUME_RELATIVE_PATH,
    RESUME_REPAIR_ID,
    ROUTE_CONTRACT_SHA256,
    ROUTE_ID,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TARGET_ROUTE_PROFILE,
    EXPECTED_CANDIDATE_FUNNEL,
    PHASE0_POLICY,
    PHASE0_SHORTLIST_SIZE,
    SELECTOR_COMPILE_COST_POLICY,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
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
    relative = safe_relative_path(value, label=label)
    result = PACKAGE_DIR / relative
    try:
        result.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped the package.") from exc
    return result


def _verify_binding(
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    binding = _mapping(raw, label=f"{label} binding")
    path = _package_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _load_job(
    job_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status")
        != "passed_inert_two_authenticated_continuations"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("execution_target") != "chtc"
        or manifest.get("row_count") != CONTINUATION_ROW_COUNT
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
    ):
        raise PackageContractError("Package manifest identity drifted.")

    resolved_job = job_path.resolve()
    matches = [
        row
        for row in _sequence(manifest.get("jobs"), label="job bindings")
        if isinstance(row, Mapping)
        and _package_path(row.get("path"), label="job path").resolve()
        == resolved_job
    ]
    if len(matches) != 1:
        raise PackageContractError("Requested job is outside the package.")
    _bound_job, job_payload = _verify_binding(
        matches[0], label="job", canonical=True
    )
    assert job_payload is not None
    job = job_payload
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("bundle_id") != BUNDLE_ID
        or job.get("execution_id") not in expected_execution_ids()
        or job.get("algorithm_id") != ALGORITHM_ID
        or job.get("route_id") != ROUTE_ID
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("route_profile") != TARGET_ROUTE_PROFILE
        or job.get("candidate_adapter_id") != CANDIDATE_ADAPTER_ID
        or job.get("candidate_representation")
        != CANDIDATE_REPRESENTATION
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or int(job.get("source_horizon", -1)) != SOURCE_HORIZON
        or int(job.get("target_horizon", -1)) != TARGET_HORIZON
        or not 0 < int(job.get("resume_round", -1)) < TARGET_HORIZON
        or job.get("resources") != RESOURCE_ENVELOPE
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Continuation job contract drifted.")

    protocol_matches = [
        row
        for row in _sequence(manifest.get("protocols"), label="protocols")
        if isinstance(row, Mapping)
        and row.get("execution_id") == job["execution_id"]
    ]
    if len(protocol_matches) != 1:
        raise PackageContractError("Job protocol binding is not unique.")
    protocol_path, protocol_payload = _verify_binding(
        protocol_matches[0], label="protocol", canonical=True
    )
    assert protocol_payload is not None
    if (
        protocol_path
        != _package_path(job["protocol"]["path"], label="protocol path")
        or protocol_matches[0].get("canonical_sha256")
        != job.get("protocol_sha256")
        or job.get("protocol") != protocol_matches[0]
    ):
        raise PackageContractError("Job protocol binding drifted.")

    _bundle_path, bundle = _verify_binding(
        manifest.get("bundle_manifest"),
        label="bundle manifest",
        canonical=True,
    )
    _locks_path, source_locks = _verify_binding(
        manifest.get("source_locks"), label="source locks", canonical=True
    )
    _composition_path, composition = _verify_binding(
        manifest.get("runtime_source_composition"),
        label="runtime source composition",
        canonical=True,
    )
    assert bundle is not None and source_locks is not None
    assert composition is not None
    if (
        bundle.get("bundle_id") != BUNDLE_ID
        or bundle.get("cell_count") != CONTINUATION_ROW_COUNT
        or bundle.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or bundle.get("source_locks_sha256") != source_locks.get("sha256")
        or bundle.get("runtime_source_composition_sha256")
        != composition.get("sha256")
        or job.get("runtime_source_composition_sha256")
        != composition.get("sha256")
    ):
        raise PackageContractError("Continuation source authority drifted.")

    resume_manifest_path, resume_manifest = _verify_binding(
        job.get("resume_manifest"), label="resume manifest", canonical=True
    )
    resume_archive_path, _ = _verify_binding(
        job.get("resume_archive"), label="resume archive"
    )
    _checkpoint_validation_path, checkpoint_validation = _verify_binding(
        job.get("checkpoint_validation"),
        label="checkpoint validation receipt",
        canonical=True,
    )
    assert resume_manifest is not None
    assert checkpoint_validation is not None
    if (
        resume_manifest_path.parent != resume_archive_path.parent
        or resume_manifest.get("archive") != job.get("resume_archive")
        or resume_manifest.get("checkpoint_sha256")
        != job.get("checkpoint_sha256")
        or resume_manifest.get("resume_round") != job.get("resume_round")
        or resume_manifest.get("target_round") != TARGET_HORIZON
        or resume_manifest.get("checkpoint_validation")
        != job.get("checkpoint_validation")
        or checkpoint_validation.get("archive")
        != job.get("resume_archive")
        or checkpoint_validation.get("members")
        != resume_manifest.get("members")
        or checkpoint_validation.get(
            "accepted_state_resume_semantic_replay_required"
        )
        is not True
        or checkpoint_validation.get("ambient_ijson_required") is not False
    ):
        raise PackageContractError("Resume input authority drifted.")
    return (
        job,
        manifest,
        protocol_payload,
        source_locks,
        composition,
        {
            "manifest": resume_manifest,
            "checkpoint_validation": checkpoint_validation,
        },
    )


def _validate_authorization(
    path: Path,
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = load_json(path, label="execution authorization")
    verify_self_digest(authorization, label="execution authorization")
    request = load_json(
        PACKAGE_DIR / "activation/activation_request.json",
        label="activation request",
    )
    verify_self_digest(request, label="activation request")
    expected_path = (
        PACKAGE_DIR
        / "activation/authorizations"
        / f"{job['execution_id']}.json"
    ).resolve()
    if (
        path.resolve() != expected_path
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != job.get("execution_id")
        or authorization.get("package_manifest_sha256")
        != manifest.get("sha256")
        or authorization.get("activation_request_sha256")
        != request.get("sha256")
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("protocol_sha256")
        != job.get("protocol_sha256")
        or authorization.get("resume_archive_sha256")
        != job.get("resume_archive", {}).get("sha256")
        or authorization.get("checkpoint_sha256")
        != job.get("checkpoint_sha256")
        or authorization.get("checkpoint_validation_sha256")
        != job.get("checkpoint_validation", {}).get("canonical_sha256")
        or authorization.get("runtime_source_composition_sha256")
        != job.get("runtime_source_composition_sha256")
        or authorization.get("pinned_image_sha256")
        != REMOTE_IMAGE_SHA256
        or authorization.get("authorization_kind")
        != "explicit_user_chtc_execution_and_submission_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("submitted") is not False
        or request.get("package_manifest_sha256") != manifest.get("sha256")
        or request.get("requested_execution_ids")
        != list(expected_execution_ids())
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("authorization_kind")
        != "explicit_user_chtc_execution_and_submission_authority"
        or request.get("paper_evidence_adoption_authorized") is not False
        or request.get("submitted") is not False
    ):
        raise PackageContractError("Execution authorization drifted.")
    return authorization


def _extract_source(
    *,
    composition: Mapping[str, Any],
    source_locks: Mapping[str, Any],
    destination: Path,
) -> None:
    archive_path, _ = _verify_binding(
        composition.get("base_archive"), label="base source archive"
    )
    _manifest_path, source_manifest = _verify_binding(
        composition.get("base_archive_manifest"),
        label="base source manifest",
        canonical=True,
    )
    assert source_manifest is not None
    if (
        composition.get("no_ambient_repo_imports") is not True
        or archive_path.stat().st_size
        != int(source_manifest.get("archive", {}).get("size_bytes", -1))
        or sha256_file(archive_path) != BASE_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("archive", {}).get("sha256")
        != BASE_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("implementation_source_inventory_sha256")
        != source_locks.get("implementation_sources", {}).get("sha256")
    ):
        raise PackageContractError("Base source archive authority drifted.")
    rows = _sequence(source_manifest.get("members"), label="source members")
    members = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(members) != len(rows) or len(rows) != int(
        source_manifest.get("member_count", -1)
    ):
        raise PackageContractError("Source member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label="source tar member"
            ).as_posix()
            row = members.get(relative)
            if (
                row is None
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(f"Unsafe source member: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(f"Unreadable source member: {relative}")
            target = destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(f"Extracted source drifted: {relative}")
            observed.add(relative)
    if observed != set(members):
        raise PackageContractError("Source archive extraction is incomplete.")

    overlays = _sequence(
        composition.get("operational_overlays"),
        label="operational overlays",
    )
    expected_overlays = (
        (
            CONTROLLER_REPAIR_ID,
            CONTROLLER_RELATIVE_PATH,
            CONTROLLER_BEFORE_SHA256,
            CONTROLLER_AFTER_SHA256,
        ),
        (
            RESUME_REPAIR_ID,
            RESUME_RELATIVE_PATH,
            RESUME_BEFORE_SHA256,
            RESUME_AFTER_SHA256,
        ),
    )
    if len(overlays) != len(expected_overlays):
        raise PackageContractError("Operational overlay closure drifted.")
    for raw_overlay, expected in zip(
        overlays, expected_overlays, strict=True
    ):
        repair_id, relative, before_sha256, after_sha256 = expected
        overlay = _mapping(raw_overlay, label="operational overlay")
        overlay_binding = _mapping(
            overlay.get("after"), label="overlay file"
        )
        overlay_path = _package_path(
            overlay_binding.get("path"), label="overlay file path"
        )
        target = destination / relative
        if (
            overlay.get("repair_id") != repair_id
            or overlay.get("path") != relative
            or overlay.get("before_sha256") != before_sha256
            or overlay_binding.get("sha256") != after_sha256
            or sha256_file(target) != before_sha256
            or not overlay_path.is_file()
            or overlay_path.is_symlink()
            or overlay_path.stat().st_size
            != int(overlay_binding.get("size_bytes", -1))
            or sha256_file(overlay_path) != after_sha256
            or overlay.get("scientific_protocol_changed") is not False
            or overlay.get("scientific_settings_changed") != []
        ):
            raise PackageContractError(
                f"Operational source overlay drifted: {repair_id}"
            )
        temporary = target.with_name(f".{target.name}.overlay")
        shutil.copyfile(overlay_path, temporary)
        os.replace(temporary, target)
        if sha256_file(target) != after_sha256:
            raise PackageContractError(
                f"Operational source overlay failed: {repair_id}"
            )


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    for module_name, expected_sha256 in (
        (
            "pipelines.static_adapt.sr_snake._controller",
            CONTROLLER_AFTER_SHA256,
        ),
        ("pipelines.static_adapt.sr_snake._resume", RESUME_AFTER_SHA256),
    ):
        module = importlib.import_module(module_name)
        try:
            module_path = Path(str(module.__file__)).resolve()
            module_path.relative_to(root)
        except ValueError as exc:
            raise PackageContractError(
                f"Runtime module escaped the source archive: {module_name}"
            ) from exc
        if sha256_file(module_path) != expected_sha256:
            raise PackageContractError(
                f"Runtime source overlay is inactive: {module_name}"
            )


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

    receipt = protocol.problem
    return resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _load_protocol(
    *,
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    source_locks: Mapping[str, Any],
) -> tuple[Any, Any]:
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    receipt = protocol.bundle_materialization
    route = protocol.route_contract
    execution = route.get("execution_settings")
    invariants = route.get("semantic_invariants")
    if (
        receipt is None
        or receipt.bundle_id != BUNDLE_ID
        or receipt.cell_id != job.get("execution_id")
        or receipt.source_locks_sha256 != source_locks.get("sha256")
        or protocol.sha256 != job.get("protocol_sha256")
        or protocol.algorithm_id != ALGORITHM_ID
        or protocol.adapter_id != CANDIDATE_ADAPTER_ID
        or protocol.candidate_representation != CANDIDATE_REPRESENTATION
        or protocol.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or protocol.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or protocol.route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
        or protocol.route_contract.get("route_profile")
        != TARGET_ROUTE_PROFILE
        or int(protocol.horizon) != TARGET_HORIZON
        or protocol.request.execution.stop.maximum_controller_rounds
        != TARGET_HORIZON
        or protocol.request.execution.resume.kind != "fresh_start"
        or protocol.request.method.admission.kind != "singleton"
        or protocol.request.method.insertion.kind != "plateau_commutation"
        or protocol.request.method.pruning.kind != "off"
        or protocol.request.method.beam.kind != "off"
        or not isinstance(execution, Mapping)
        or execution.get("phase3_backend_cost_scope")
        != BACKEND_COMPILE_SCOPE
        or execution.get("static_lane_route") != "global_single_population"
        or "physical_lane_shortlist_aggressiveness" in execution
        or not isinstance(invariants, Mapping)
        or invariants.get("selector_compile_cost_policy")
        != SELECTOR_COMPILE_COST_POLICY
        or invariants.get("phase_i_compile_cost_source")
        != "structural_proxy_v1"
        or invariants.get("phase_ii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get("phase_iii_compile_cost_source")
        != "backend_transpile_v1"
        or invariants.get("physical_operator_lanes_active") is not False
        or invariants.get("shortlist_population_policy")
        != "single_global_population_v1"
        or invariants.get("plateau_prior_mean_decrease_ratio_threshold")
        != 1.0e-4
        or invariants.get("candidate_funnel_order")
        != EXPECTED_CANDIDATE_FUNNEL
        or execution.get("ra_phase0_gradient_shortlist_policy")
        != PHASE0_POLICY
        or execution.get("ra_phase0_gradient_shortlist_size")
        != PHASE0_SHORTLIST_SIZE
    ):
        raise PackageContractError("Typed Page-12 continuation protocol drifted.")
    authority = _mint_bundle_protocol_materialization_authority(
        receipt,
        source_lock_refs=protocol.source_locks,
        protocol_sha256=protocol.sha256,
    )
    protocol = _attach_validated_bundle_protocol_authority(protocol, authority)
    return protocol, _problem_from_protocol(protocol)


def _extract_resume(
    *,
    job: Mapping[str, Any],
    resume_context: Mapping[str, Any],
    destination: Path,
) -> Path:
    archive_path = _package_path(
        job["resume_archive"]["path"], label="resume archive"
    )
    manifest = _mapping(
        resume_context.get("manifest"), label="resume manifest"
    )
    checkpoint_validation = _mapping(
        resume_context.get("checkpoint_validation"),
        label="checkpoint validation receipt",
    )
    validated = validate_resume_archive(
        archive_path,
        manifest,
        expected_round=int(job["resume_round"]),
        checkpoint_validation=checkpoint_validation,
        verify_archive_members=False,
    )
    rows = {
        str(row["path"]): row
        for row in manifest["members"]
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
            digest = hashlib.sha256()
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
    checkpoint = destination / Path(
        str(validated["members_by_role"]["checkpoint"]["path"])
    )
    if sha256_file(checkpoint) != job.get("checkpoint_sha256"):
        raise PackageContractError("Extracted checkpoint identity drifted.")
    return checkpoint


def _prepare(
    job_path: Path,
    *,
    verify_resume_bytes: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    Any,
    dict[str, Any],
    tempfile.TemporaryDirectory[str],
]:
    (
        job,
        manifest,
        protocol_payload,
        source_locks,
        composition,
        resume_context,
    ) = _load_job(job_path)
    temporary = tempfile.TemporaryDirectory(
        prefix=f"paper-i-page12-r70-{job['regime_id']}."
    )
    try:
        root = Path(temporary.name)
        source_root = root / "source"
        _extract_source(
            composition=composition,
            source_locks=source_locks,
            destination=source_root,
        )
        original = Path.cwd()
        os.chdir(source_root)
        try:
            _activate_source_root(source_root)
            protocol, problem = _load_protocol(
                job=job,
                payload=protocol_payload,
                source_locks=source_locks,
            )
        finally:
            os.chdir(original)
        if verify_resume_bytes:
            validate_resume_archive(
                _package_path(
                    job["resume_archive"]["path"], label="resume archive"
                ),
                resume_context["manifest"],
                expected_round=int(job["resume_round"]),
                checkpoint_validation=resume_context[
                    "checkpoint_validation"
                ],
            )
    except BaseException:
        temporary.cleanup()
        raise
    return job, manifest, protocol, problem, resume_context, temporary


def preflight(job_path: Path, *, verify_resume_bytes: bool) -> dict[str, Any]:
    job, manifest, protocol, _problem, _resume, temporary = _prepare(
        job_path, verify_resume_bytes=verify_resume_bytes
    )
    temporary.cleanup()
    return digested(
        {
            "schema": "paper_i_page12_strong_r70_worker_preflight_v2",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": protocol.route_contract["sha256"],
            "resume_round": job["resume_round"],
            "target_horizon": TARGET_HORIZON,
            "accepted_energy_roundoff_overlay_active": True,
            "resume_bytes_verified": verify_resume_bytes,
            "scientific_execution_performed": False,
        }
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _publish_staging(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise PackageContractError("Worker output destination already exists.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sibling = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.publish-", dir=destination.parent
        )
    )
    try:
        shutil.copytree(staging, sibling, dirs_exist_ok=True)
        os.rename(sibling, destination)
    except BaseException:
        shutil.rmtree(sibling, ignore_errors=True)
        raise


def _expected_artifact_paths(job: Mapping[str, Any]) -> dict[str, str]:
    execution = str(job["execution_id"])
    root = PurePosixPath("runs") / execution
    expected = {
        "execution_manifest": "execution_manifest.json",
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
    }
    raw = _mapping(job.get("expected_artifacts"), label="expected artifacts")
    for role, suffix in expected.items():
        if raw.get(role) != (root / suffix).as_posix():
            raise PackageContractError(f"Expected {role} path drifted.")
    return expected


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    job, manifest, protocol, problem, resume_context, temporary = _prepare(
        job_path, verify_resume_bytes=False
    )
    try:
        authority = _validate_authorization(
            authorization_path, job=job, manifest=manifest
        )
        if (
            output_dir.exists()
            or output_dir.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
            or output_dir.name != job["execution_id"]
            or output_dir.parent.name != "runs"
        ):
            raise PackageContractError("Worker destination is unsafe.")
        root = Path(temporary.name)
        source_root = root / "source"
        resume_root = root / "resume_input"
        checkpoint = _extract_resume(
            job=job,
            resume_context=resume_context,
            destination=resume_root,
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

        controls = RAAdaptOperationalControls(
            maximum_controller_rounds=TARGET_HORIZON,
            resume=AcceptedStateResume(
                checkpoint_path=checkpoint,
                checkpoint_sha256=str(job["checkpoint_sha256"]),
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
                resource_rounds=(TARGET_HORIZON,),
            ),
        )
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result = run_ra_adapt(
                problem, protocol, operational_controls=controls
            )
        finally:
            os.chdir(original)
        rounds = len(result.run.accepted_trajectory)
        source_metadata = _mapping(
            resume_context["checkpoint_validation"].get("metadata"),
            label="source checkpoint metadata",
        )
        accepted_prefix = result.run.accepted_trajectory[:SOURCE_HORIZON]
        terminal_prefix = accepted_prefix[-1] if accepted_prefix else None
        if (
            len(accepted_prefix) != SOURCE_HORIZON
            or terminal_prefix is None
            or terminal_prefix.controller_round != SOURCE_HORIZON
            or not math.isclose(
                float(terminal_prefix.energy),
                float(source_metadata["accepted_prefix_terminal_energy"]),
                rel_tol=0.0,
                abs_tol=(
                    128.0
                    * math.ulp(
                        max(
                            1.0,
                            abs(float(terminal_prefix.energy)),
                            abs(
                                float(
                                    source_metadata[
                                        "accepted_prefix_terminal_energy"
                                    ]
                                )
                            ),
                        )
                    )
                ),
            )
            or terminal_prefix.projective_state_fingerprint
            != source_metadata[
                "accepted_prefix_terminal_state_fingerprint"
            ]
        ):
            raise PackageContractError(
                "Authenticated first-50 accepted-state prefix changed."
            )
        if (
            result.protocol.sha256 != protocol.sha256
            or rounds != TARGET_HORIZON
            or not (staging / "checkpoints/current.json").is_file()
            or not (staging / "result/estimator_ledger.json").is_file()
        ):
            raise PackageContractError(
                f"Continuation stopped at round {rounds}, not {TARGET_HORIZON}."
            )
        _write_json(staging / "result/result.json", result.to_dict())
        if result.run.paper_i_summary is None:
            raise PackageContractError("Page-12 continuation summary is absent.")
        _write_json(
            staging / "summary/summary.json",
            result.run.paper_i_summary.to_dict(),
        )
        expected = _expected_artifact_paths(job)
        payloads = {
            role: {
                "path": job["expected_artifacts"][role],
                "sha256": sha256_file(staging / suffix),
                "size_bytes": (staging / suffix).stat().st_size,
            }
            for role, suffix in expected.items()
            if role != "execution_manifest"
        }
        execution_manifest = digested(
            {
                "schema": "paper_i_page12_strong_r70_execution_manifest_v2",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "resume_round": job["resume_round"],
                "target_horizon": TARGET_HORIZON,
                "controller_rounds_completed": rounds,
                "source_checkpoint_sha256": job["checkpoint_sha256"],
                "accepted_state_resume": True,
                "accepted_energy_roundoff_overlay": CONTROLLER_REPAIR_ID,
                "operational_source_overlays": [
                    CONTROLLER_REPAIR_ID,
                    RESUME_REPAIR_ID,
                ],
                "accepted_prefix_preservation": {
                    "status": "passed",
                    "source_round": SOURCE_HORIZON,
                    "source_checkpoint_sha256": job["checkpoint_sha256"],
                    "terminal_energy": float(terminal_prefix.energy),
                    "terminal_state_fingerprint": (
                        terminal_prefix.projective_state_fingerprint
                    ),
                },
                "output_payloads": payloads,
            }
        )
        _write_json(
            staging / expected["execution_manifest"], execution_manifest
        )
        if any(not (staging / suffix).is_file() for suffix in expected.values()):
            raise PackageContractError("Expected output closure is incomplete.")
        _publish_staging(staging, output_dir)
        receipt = digested(
            {
                "schema": "paper_i_page12_strong_r70_worker_receipt_v2",
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "resume_round": job["resume_round"],
                "controller_rounds_completed": rounds,
                "accepted_state_resume": True,
                "operational_source_overlays": [
                    CONTROLLER_REPAIR_ID,
                    RESUME_REPAIR_ID,
                ],
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--verify-resume-bytes", action="store_true")
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
            payload = preflight(
                args.job.resolve(),
                verify_resume_bytes=args.verify_resume_bytes,
            )
        else:
            if args.verify_resume_bytes:
                raise PackageContractError(
                    "Execution always verifies resume bytes internally."
                )
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
    except (
        OSError,
        PackageContractError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
