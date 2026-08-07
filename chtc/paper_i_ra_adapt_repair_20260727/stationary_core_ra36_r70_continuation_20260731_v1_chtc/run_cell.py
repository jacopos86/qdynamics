#!/usr/bin/env python3
"""Run one explicitly authorized r70 continuation cell.

This worker is intentionally not wired to HTCondor.  Fresh always-insertion
rows fail closed while their exact round-50 predecessors remain collision
blocked.  Resume rows require a separate, per-cell authorization document that
is not shipped with this inert package.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    COLLISION_STATUS_NAME,
    EXECUTION_PLAN_NAME,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_MANIFEST_SCHEMA,
    RESOURCE_WEIGHTING_SCOPE,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    digested,
    load_json,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


DERIVED_CHANGED_PATHS = (
    "horizon",
    "request.execution.stop.maximum_controller_rounds",
    "sha256",
    "stopping_rule.maximum_controller_rounds",
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
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(
            f"{label} escaped the package."
        ) from exc
    return path


def _verify_local_binding(
    binding: Mapping[str, Any], *, label: str
) -> Path:
    path = _package_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("sha256")
    ):
        raise PackageContractError(f"{label} binding drifted.")
    return path


def _load_runtime_job(job_path: Path) -> dict[str, Any]:
    manifest = load_json(
        PACKAGE_DIR / PACKAGE_MANIFEST_NAME,
        label="package manifest",
    )
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("status")
        != "passed_inert_collision_blocked"
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("authority_overlay_present") is not False
    ):
        raise PackageContractError(
            "Runtime package is not the bound inert package."
        )
    if (PACKAGE_DIR / "submit.sub").exists() or (
        PACKAGE_DIR / "authority"
    ).exists():
        raise PackageContractError(
            "The inert package gained submit or authority state."
        )
    plan_binding = _mapping(
        manifest.get("execution_plan"),
        label="execution-plan binding",
    )
    collision_binding = _mapping(
        manifest.get("collision_status"),
        label="collision-status binding",
    )
    for binding, name, label in (
        (plan_binding, EXECUTION_PLAN_NAME, "execution plan"),
        (
            collision_binding,
            COLLISION_STATUS_NAME,
            "collision status",
        ),
    ):
        if binding.get("path") != name:
            raise PackageContractError(f"{label} path drifted.")
        path = PACKAGE_DIR / name
        payload = load_json(path, label=label)
        digest = verify_self_digest(payload, label=label)
        if (
            digest != binding.get("sha256")
            or sha256_file(path) != binding.get("file_sha256")
        ):
            raise PackageContractError(f"{label} binding drifted.")
    plan = load_json(PACKAGE_DIR / EXECUTION_PLAN_NAME, label="plan")
    collision = load_json(
        PACKAGE_DIR / COLLISION_STATUS_NAME,
        label="collision",
    )
    if (
        plan.get("submission_ready") is not False
        or plan.get("execution_authorized") is not False
        or collision.get("blocking") is not True
        or collision.get("may_submit") is not False
    ):
        raise PackageContractError(
            "The package-wide collision/submission gate drifted."
        )
    job = load_json(job_path, label="job")
    job_digest = verify_self_digest(job, label="job")
    execution_id = str(job.get("execution_id", ""))
    expected_path = (
        PACKAGE_DIR / "jobs" / f"{execution_id}.json"
    ).resolve()
    job_rows = {
        str(row["execution_id"]): row
        for row in _sequence(
            manifest.get("jobs"), label="job bindings"
        )
        if isinstance(row, Mapping)
    }
    binding = _mapping(
        job_rows.get(execution_id), label="job manifest binding"
    )
    if (
        job_path.resolve() != expected_path
        or binding.get("path")
        != f"jobs/{execution_id}.json"
        or binding.get("canonical_sha256") != job_digest
        or binding.get("sha256") != sha256_file(job_path)
        or int(binding.get("size_bytes", -1))
        != job_path.stat().st_size
        or job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("active_gradient_policy")
        != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope")
        != RESOURCE_WEIGHTING_SCOPE
        or int(job.get("source_horizon", -1))
        != SOURCE_HORIZON
        or int(job.get("target_horizon", -1))
        != TARGET_HORIZON
        or job.get("global_submission_blocked") is not True
        or job.get("submission_ready") is not False
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Worker job identity drifted.")
    return job


def _validate_authorization(
    path: Path, *, job: Mapping[str, Any]
) -> dict[str, Any]:
    authorization = load_json(path, label="execution authorization")
    verify_self_digest(
        authorization, label="execution authorization"
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id")
        != job.get("execution_id")
        or authorization.get("job_spec_sha256")
        != job.get("sha256")
        or authorization.get("scope")
        != "single_cell_execution_only"
        or authorization.get("authorization_kind")
        != "explicit_user_execution_authority"
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get(
            "global_submission_block_acknowledged"
        )
        is not True
        or authorization.get("collision_status_sha256")
        != job.get("collision_status_sha256")
    ):
        raise PackageContractError(
            "Execution authorization is absent, stale, or out of scope."
        )
    return authorization


def _extract_bound_tar(
    *,
    archive_path: Path,
    members: Mapping[str, Mapping[str, Any]],
    destination: Path,
    label: str,
) -> None:
    observed: set[str] = set()
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            binding = members.get(member.name)
            relative = safe_relative_path(
                member.name, label=f"{label} member path"
            )
            if (
                binding is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size
                != int(binding.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"{label} contains an unexpected member: "
                    f"{member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(
                    f"{label} member is unreadable: {member.name}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(
                    lambda: source.read(1024 * 1024), b""
                ):
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
                    f"{label} member digest drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(members):
        raise PackageContractError(
            f"{label} member closure drifted."
        )


def _extract_source(job: Mapping[str, Any], destination: Path) -> None:
    archive_binding = _mapping(
        job.get("source_archive"), label="source archive binding"
    )
    manifest_binding = _mapping(
        job.get("source_archive_manifest"),
        label="source archive manifest binding",
    )
    archive = _verify_local_binding(
        archive_binding, label="source archive"
    )
    manifest_path = _verify_local_binding(
        manifest_binding, label="source archive manifest"
    )
    manifest = load_json(
        manifest_path, label="source archive manifest"
    )
    verify_self_digest(manifest, label="source archive manifest")
    manifest_archive = _mapping(
        manifest.get("archive"), label="manifest archive"
    )
    member_rows = _sequence(
        manifest.get("members"), label="source archive members"
    )
    members = {
        safe_relative_path(
            row.get("path"), label="source member path"
        ).as_posix(): _mapping(row, label="source member")
        for row in member_rows
        if isinstance(row, Mapping)
    }
    if (
        manifest_archive.get("sha256")
        != archive_binding.get("sha256")
        or int(manifest_archive.get("size_bytes", -1))
        != int(archive_binding.get("size_bytes", -1))
        or len(members)
        != len(member_rows)
        != int(manifest.get("member_count", -1))
    ):
        raise PackageContractError(
            "Source archive manifest closure drifted."
        )
    _extract_bound_tar(
        archive_path=archive,
        members=members,
        destination=destination,
        label="source archive",
    )


def _extract_resume(
    job: Mapping[str, Any], destination: Path
) -> Path:
    resume = _mapping(
        job.get("resume_input"), label="resume input"
    )
    archive = _verify_local_binding(
        _mapping(
            resume.get("archive"), label="resume archive binding"
        ),
        label="resume archive",
    )
    member_rows = _sequence(
        resume.get("members"), label="resume members"
    )
    members = {
        safe_relative_path(
            row.get("path"), label="resume member path"
        ).as_posix(): _mapping(row, label="resume member")
        for row in member_rows
        if isinstance(row, Mapping)
    }
    checkpoint_relative = safe_relative_path(
        resume.get("checkpoint_path"),
        label="resume checkpoint path",
    )
    if (
        len(members) != len(member_rows)
        or len(member_rows)
        != int(resume.get("member_count", -1))
        or len(members) != 3
        or resume.get("pointer_closed") is not True
        or checkpoint_relative.as_posix() not in members
        or members[checkpoint_relative.as_posix()].get("sha256")
        != resume.get("checkpoint_sha256")
    ):
        raise PackageContractError(
            "Compact resume-input closure drifted."
        )
    _extract_bound_tar(
        archive_path=archive,
        members=members,
        destination=destination,
        label="resume archive",
    )
    checkpoint = destination / checkpoint_relative
    if sha256_file(checkpoint) != resume.get("checkpoint_sha256"):
        raise PackageContractError(
            "Extracted resume checkpoint digest drifted."
        )
    return checkpoint


def _module_is_under(module: ModuleType, root: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is not None:
        try:
            Path(origin).resolve().relative_to(root)
            return True
        except ValueError:
            return False
    locations = getattr(module, "__path__", None)
    if locations is None:
        return True
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
        except ValueError:
            return False
    return True


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    if not root.is_dir():
        raise PackageContractError(
            f"Extracted source root is unavailable: {root}"
        )
    drifted: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            continue
        if not _module_is_under(module, root):
            drifted.append(name)
    if drifted:
        raise PackageContractError(
            "Project modules were imported before source-lock activation: "
            + ", ".join(sorted(drifted))
        )
    source_text = str(root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        ResolvedProblemReceipt,
    )

    receipt = protocol.problem
    problem = resolve_problem_context(
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
    if (
        ResolvedProblemReceipt.from_problem(problem).to_dict()
        != receipt.to_dict()
    ):
        raise PackageContractError(
            "Reconstructed problem drifted from the source protocol."
        )
    return problem


def _scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return [(path, before, after)]
        rows: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(before):
            rows.extend(
                _scalar_differences(
                    before[key],
                    after[key],
                    path=(*path, str(key)),
                )
            )
        return rows
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [(path, before, after)]
        rows = []
        for index, (left, right) in enumerate(zip(before, after)):
            rows.extend(
                _scalar_differences(
                    left, right, path=(*path, index)
                )
            )
        return rows
    return [] if before == after else [(path, before, after)]


def _derived_protocol(
    *, job: Mapping[str, Any], source_root: Path
) -> tuple[Any, Any, dict[str, Any]]:
    from pipelines.static_adapt.ra_adapt.bundles import (
        BundleCellSpec,
        _as_protocol_payload,
        _build_request,
        _bundle_protocol_materialization_authority,
        _decorate_protocol_payload,
        _source_lock_refs,
        _validate_protocol_payload,
        load_validated_bundle_protocol,
    )
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )
    from pipelines.static_adapt.ra_adapt.engine import (
        build_resolved_ra_protocol,
    )
    from pipelines.static_adapt.sr_snake.contracts import FreshStart

    source_binding = _mapping(
        job.get("source_protocol"), label="source protocol binding"
    )
    relative = safe_relative_path(
        source_binding.get("path"),
        label="source protocol path",
    )
    source_path = source_root / relative
    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_size
        != int(source_binding.get("size_bytes", -1))
        or sha256_file(source_path) != source_binding.get("sha256")
    ):
        raise PackageContractError(
            "Extracted source protocol byte binding drifted."
        )
    source = load_validated_bundle_protocol(source_path)
    if (
        source.sha256
        != source_binding.get("canonical_sha256")
        or int(source.horizon) != SOURCE_HORIZON
        or source.active_gradient_policy
        != ACTIVE_GRADIENT_POLICY
        or source.resource_weighting_scope
        != RESOURCE_WEIGHTING_SCOPE
        or not isinstance(source.request.execution.resume, FreshStart)
    ):
        raise PackageContractError(
            "Source protocol is not the bound stationary/late r50 row."
        )
    source_materialization = source.bundle_materialization
    if (
        source_materialization is None
        or source_materialization.cell_id
        != job.get("source_execution_id")
        or source_materialization.source_lock_id
        != source.source_locks.get("cell_source_lock_id")
    ):
        raise PackageContractError(
            "Source protocol materialization identity drifted."
        )
    bundle_root = source_path.parent.parent
    manifest = load_json(
        bundle_root / "bundle_manifest.json",
        label="source bundle manifest",
    )
    source_locks = load_json(
        bundle_root / "source_locks.json",
        label="source locks",
    )
    verify_self_digest(manifest, label="source bundle manifest")
    verify_self_digest(source_locks, label="source locks")
    cell = BundleCellSpec(
        cell_id=str(source_materialization.cell_id),
        stage="core",
        regime_id=str(job["regime_id"]),
        nph=int(job["nph"]),
        route_id=str(job["route_id"]),
        algorithm_id=str(source.algorithm_id),
        selector_family="ra_adapt",
        candidate_representation=str(
            job["candidate_representation"]
        ),
        horizon=TARGET_HORIZON,
        source_lock_id=str(source_materialization.source_lock_id),
    )
    lock_refs = _source_lock_refs(source_locks, cell=cell)
    cell_lock = _mapping(
        _mapping(
            source_locks.get("cell_locks"),
            label="cell source locks",
        ).get(cell.source_lock_id),
        label="cell source lock",
    )
    request = _build_request(cell, bundle_dir=bundle_root)
    if not isinstance(request.execution.resume, FreshStart):
        raise PackageContractError(
            "Derived protocol did not retain fresh-start authority."
        )
    authority_kwargs = {
        "cell": cell,
        "bundle_id": str(source.bundle_id),
        "bundle_manifest_sha256": str(manifest["sha256"]),
        "source_locks_sha256": str(source_locks["sha256"]),
        "source_lock_refs": lock_refs,
        "active_gradient_policy": str(
            source.active_gradient_policy
        ),
        "resource_weighting_scope": str(
            source.resource_weighting_scope
        ),
    }
    initial_authority = (
        _bundle_protocol_materialization_authority(
            **authority_kwargs
        )
    )
    resolved = build_resolved_ra_protocol(
        _problem_from_protocol(source),
        request,
        materialization_authority=initial_authority,
    )
    payload = _as_protocol_payload(resolved, cell=cell)
    payload = _decorate_protocol_payload(
        payload,
        cell=cell,
        request=request,
        cell_source_lock=cell_lock,
        materialization_authority=initial_authority,
    )
    validation_kwargs = {
        "cell": cell,
        "bundle_id": str(source.bundle_id),
        "bundle_manifest_sha256": str(manifest["sha256"]),
        "active_gradient_policy": str(
            source.active_gradient_policy
        ),
        "resource_weighting_scope": str(
            source.resource_weighting_scope
        ),
        "source_lock_refs": lock_refs,
        "cell_source_lock": cell_lock,
        "source_locks_sha256": str(source_locks["sha256"]),
    }
    accepted = inspect.signature(
        _validate_protocol_payload
    ).parameters
    _validate_protocol_payload(
        payload,
        **{
            key: value
            for key, value in validation_kwargs.items()
            if key in accepted
        },
    )
    derived = resolved_ra_adapt_protocol_from_mapping(payload)
    final_authority = _bundle_protocol_materialization_authority(
        **authority_kwargs,
        protocol_sha256=derived.sha256,
    )
    derived = _attach_validated_bundle_protocol_authority(
        derived, final_authority
    )
    changed = sorted(
        ".".join(str(component) for component in path)
        for path, _before, _after in _scalar_differences(
            source.to_dict(), derived.to_dict()
        )
    )
    if (
        changed != list(DERIVED_CHANGED_PATHS)
        or int(derived.horizon) != TARGET_HORIZON
        or derived.active_gradient_policy
        != source.active_gradient_policy
        or derived.resource_weighting_scope
        != source.resource_weighting_scope
        or derived.route_contract != source.route_contract
        or derived.source_locks != source.source_locks
        or derived.problem != source.problem
        or derived.parent_inventory != source.parent_inventory
        or derived.executable_pool != source.executable_pool
        or derived.bundle_materialization
        != source.bundle_materialization
    ):
        raise PackageContractError(
            "Derived r70 protocol changed a non-horizon setting: "
            f"{changed}"
        )
    delta = {
        "changed_paths": changed,
        "only_scientific_change": (
            "maximum_controller_rounds_50_to_70"
        ),
        "source_protocol_sha256": source.sha256,
        "derived_protocol_sha256": derived.sha256,
        "non_swept_settings_diff": [],
        "fields_added_by_current_defaults": [],
        "stationary_gradient_preserved": True,
        "late_resource_weighting_preserved": True,
    }
    return derived, _problem_from_protocol(source), delta


def _observation_controls(
    *,
    output_root: Path,
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None,
) -> Any:
    from pipelines.static_adapt.ra_adapt.contracts import (
        RAAdaptOperationalControls,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        AcceptedStateResume,
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    resume = (
        FreshStart()
        if checkpoint_path is None
        else AcceptedStateResume(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=str(checkpoint_sha256),
        )
    )
    observation = SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=output_root / "checkpoint.json",
            every_controller_rounds=1,
            keep_history_tail=100,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=output_root / "estimator_ledger.json"
        ),
    )
    return RAAdaptOperationalControls(
        maximum_controller_rounds=TARGET_HORIZON,
        resume=resume,
        observation=observation,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        text = json.JSONEncoder(
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).iterencode(dict(payload))
        for chunk in text:
            stream.write(chunk.encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_result_artifacts(
    *,
    staging: Path,
    job: Mapping[str, Any],
    authorization: Mapping[str, Any],
    protocol: Any,
    result: Any,
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    result_payload = result.to_dict()
    if not isinstance(result_payload, Mapping):
        raise PackageContractError("RA result did not serialize.")
    summary_value = result.run.paper_i_summary
    summary = (
        summary_value.to_dict()
        if callable(getattr(summary_value, "to_dict", None))
        else None
    )
    rounds = len(result.run.accepted_trajectory)
    minimum = (
        SOURCE_HORIZON
        if job["execution_mode"]
        == "authenticated_resume_50_to_70"
        else 1
    )
    if (
        not isinstance(summary, Mapping)
        or result.protocol.sha256 != protocol.sha256
        or int(protocol.horizon) != TARGET_HORIZON
        or not minimum <= rounds <= TARGET_HORIZON
        or int(summary.get("available_controller_rounds", -1))
        != rounds
    ):
        raise PackageContractError(
            "Round-70 continuation result closure failed."
        )
    _write_json(staging / "result.json", result_payload)
    _write_json(staging / "summary.json", summary)
    expected = {
        "checkpoint": staging / "checkpoint.json",
        "estimator_ledger": staging / "estimator_ledger.json",
        "result": staging / "result.json",
        "summary": staging / "summary.json",
    }
    if any(not path.is_file() for path in expected.values()):
        raise PackageContractError(
            "Continuation output artifacts are incomplete."
        )
    execution_manifest = digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_r70_"
                "continuation_execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "execution_mode": job["execution_mode"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_protocol_sha256": delta[
                "source_protocol_sha256"
            ],
            "derived_protocol_sha256": protocol.sha256,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "controller_rounds_available": rounds,
            "only_scientific_change": delta[
                "only_scientific_change"
            ],
            "changed_protocol_paths": delta["changed_paths"],
            "non_swept_settings_diff": [],
            "stationary_gradient_preserved": True,
            "late_resource_weighting_preserved": True,
            "output_payloads": {
                role: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for role, path in sorted(expected.items())
            },
        }
    )
    _write_json(
        staging / "execution_manifest.json",
        execution_manifest,
    )
    return execution_manifest


def run_cell(
    *,
    job_path: Path,
    authorization_path: Path,
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    job = _load_runtime_job(job_path)
    if (
        job.get("execution_mode") == "fresh_0_to_70"
        or job.get("collision_status")
        == "blocked_live_r50_predecessor"
    ):
        collision = _mapping(
            job.get("collision"), label="fresh-row collision"
        )
        raise PackageContractError(
            "Fresh 0→70 row is blocked by its live exact r50 "
            f"predecessor {collision.get('cluster_id')}."
            f"{collision.get('proc_id')}; no execution or supersession "
            "is permitted."
        )
    if (
        job.get("execution_mode")
        != "authenticated_resume_50_to_70"
        or job.get("collision_status") != "none"
        or job.get("resume_input") is None
    ):
        raise PackageContractError(
            "Worker accepts only authenticated 50→70 resume rows."
        )
    authorization = _validate_authorization(
        authorization_path, job=job
    )
    if (
        output_dir.exists()
        or output_dir.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise PackageContractError(
            "Worker destination already exists."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{job['execution_id']}.",
        dir=output_dir.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        source_root = temporary / "source"
        resume_root = temporary / "resume"
        staging = temporary / "artifacts"
        staging.mkdir()
        _extract_source(job, source_root)
        checkpoint = _extract_resume(job, resume_root)
        _activate_source_root(source_root)
        protocol, problem, delta = _derived_protocol(
            job=job, source_root=source_root
        )
        controls = _observation_controls(
            output_root=staging,
            checkpoint_path=checkpoint,
            checkpoint_sha256=job["resume_input"][
                "checkpoint_sha256"
            ],
        )
        from pipelines.static_adapt.ra_adapt import run_ra_adapt

        original = Path.cwd()
        os.chdir(
            (
                source_root
                / safe_relative_path(
                    job["source_protocol"]["path"],
                    label="source protocol path",
                )
            ).parent.parent
        )
        try:
            result = run_ra_adapt(
                problem,
                protocol,
                operational_controls=controls,
            )
        finally:
            os.chdir(original)
        execution_manifest = _write_result_artifacts(
            staging=staging,
            job=job,
            authorization=authorization,
            protocol=protocol,
            result=result,
            delta=delta,
        )
        os.rename(staging, output_dir)
    receipt = digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_r70_"
                "continuation_worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": job["execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "output_dir": output_dir.as_posix(),
            "output_files": [
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(
                    output_dir.iterdir(), key=lambda item: item.name
                )
                if path.is_file()
            ],
        }
    )
    _write_json(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--execution-authorization",
        "--authorization",
        dest="authorization",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = run_cell(
            job_path=args.job.resolve(),
            authorization_path=args.authorization.resolve(),
            output_dir=args.output_dir.resolve(),
            receipt_path=args.receipt.resolve(),
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
