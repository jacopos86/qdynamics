#!/usr/bin/env python3
"""Preflight or execute one local Page-12 matched singleton cell."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import psutil
import shutil
import subprocess
import sys
import stat
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from package_contract import *  # noqa: E402,F403


ACTIVATION_SCHEMA = "paper_i_page12_matched_singleton12_local_activation_v1"
WORKER_PREFLIGHT_SCHEMA = "paper_i_page12_matched_singleton12_worker_preflight_v1"
WORKER_RECEIPT_SCHEMA = "paper_i_page12_matched_singleton12_worker_receipt_v1"
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_page12_matched_singleton12_cell_execution_manifest_v1"
)
NATIVE_RUNTIME_SCHEMA = (
    "paper_i_page12_matched_singleton12_native_local_runtime_receipt_v1"
)
DEFAULT_ACTIVATION_PATH = REPAIR_ROOT / (
    "paper_i_page12_matched_singleton12_r50_20260815_v1_activation/"
    "activation_manifest.json"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/paper_i_page12_matched_singleton12_r50_20260815_v1"
)
DEFAULT_HANDOFF_STATE_DIR = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_handoff_state_20260815_v1"
)
DEFAULT_HANDOFF_RECEIPT = DEFAULT_HANDOFF_STATE_DIR / "handoff_receipt.json"
DEFAULT_HANDOFF_LOCK = DEFAULT_HANDOFF_STATE_DIR / "handoff.lock"
DEFAULT_TARGET_CONTRACT = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_target_contract_20260815.json"
)
RUNNER_PATH = REPAIR_ROOT / (
    "run_local_paper_i_page12_matched_singleton12_r50_20260815.py"
)
HANDOFF_RECEIPT_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_RECEIPT"
HANDOFF_TOKEN_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_TOKEN"
HANDOFF_LOCK_FD_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_LOCK_FD"
CHILD_TOKEN_ENV = "PAPER_I_MATCHED_SINGLETON12_CHILD_TOKEN"
HANDOFF_RECEIPT_SCHEMA = "paper_i_matched_singleton12_handoff_receipt_v1"
HANDOFF_RECEIPT_STATUS = (
    "passed_source_terminal_and_target_activation_authorized_pending_exec"
)
REQUIRED_NUMERICAL_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
}
PACKAGE_CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
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
        raise PackageContractError(f"{label} escaped the package.") from exc
    return path


def _verify_binding(
    raw: Any, *, label: str, canonical: bool = False
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = _package_path(row.get("path"), label=f"{label} path")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != row.get("canonical_sha256"):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _verify_control_file_closure(manifest: Mapping[str, Any]) -> None:
    raw_rows = _sequence(manifest.get("control_files"), label="control file bindings")
    if len(raw_rows) != len(PACKAGE_CONTROL_FILES):
        raise PackageContractError("Package control file closure drifted.")
    rows = [
        _mapping(raw, label=f"control file {index} binding")
        for index, raw in enumerate(raw_rows)
    ]
    if [row.get("path") for row in rows] != list(PACKAGE_CONTROL_FILES):
        raise PackageContractError("Package control file inventory drifted.")
    for index, row in enumerate(rows):
        if set(row) != {"path", "sha256", "size_bytes"}:
            raise PackageContractError(
                f"Control file {index} binding is malformed."
            )
        _verify_binding(row, label=f"control file {index}", canonical=False)


def _load_closed_job(
    job_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_matched_singleton12"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 12
        or manifest.get("execution_ids") != list(expected_execution_ids())
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("paper_adoption_authorized") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
        or manifest.get("operational_checkpoint_overlay")
        != {
            "relative_path": OPERATIONAL_CHECKPOINT_OVERLAY.as_posix(),
            "sealed_sha256": SEALED_CHECKPOINT_SHA256,
            "candidate_sha256": OPERATIONAL_CHECKPOINT_OVERLAY_SHA256,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "post_extraction_overlay_count": 1,
            "ambient_resume_overlay": False,
            "sealed_resume_reader_path": SEALED_RESUME_READER.as_posix(),
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "parity_canary_scope": PARITY_CANARY_SCOPE,
            "multi_round_compact_tail_resume_validated": False,
            "scientific_parity_receipt_file_sha256": (
                STRONG5_PARITY_RECEIPT_FILE_SHA256
            ),
            "scientific_parity_receipt_canonical_sha256": (
                STRONG5_PARITY_RECEIPT_CANONICAL_SHA256
            ),
        }
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert matched singleton-12 manifest drifted.")
    _verify_control_file_closure(manifest)

    resolved_job = job_path.resolve()
    matching_jobs: list[Mapping[str, Any]] = []
    for row in _sequence(manifest.get("jobs"), label="job bindings"):
        if isinstance(row, Mapping) and _package_path(
            row.get("path"), label="job path"
        ).resolve() == resolved_job:
            matching_jobs.append(row)
    if len(matching_jobs) != 1:
        raise PackageContractError("Requested job is outside the sealed package.")
    bound_job, job_payload = _verify_binding(
        matching_jobs[0], label="job", canonical=True
    )
    assert job_payload is not None
    job = job_payload
    execution = str(job.get("execution_id", ""))
    method = str(job.get("method", ""))
    expected_algorithm = (
        RA_ALGORITHM_ID if method == "ra_singleton_plateau" else APPEND_ALGORITHM_ID
    )
    expected_adapter = RA_ADAPTER_ID if method == "ra_singleton_plateau" else APPEND_ADAPTER_ID
    expected_entrypoint = "run_ra_adapt" if method == "ra_singleton_plateau" else "run_append_adapt"
    if (
        bound_job.resolve() != resolved_job
        or execution not in expected_execution_ids()
        or job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("bundle_id") != BUNDLE_ID
        or method not in METHODS
        or job.get("algorithm_id") != expected_algorithm
        or job.get("candidate_adapter_id") != expected_adapter
        or job.get("execution_entrypoint") != expected_entrypoint
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope") != RESOURCE_WEIGHTING_SCOPE
        or job.get("candidate_representation") != CANDIDATE_REPRESENTATION
        or job.get("fresh_start_contract")
        != {
            "kind": "fresh_start",
            "source_checkpoint": None,
            "resume_archive": None,
            "fresh_start_only": True,
            "checkpoint_resume_authorized": False,
        }
        or job.get("checkpoint_observation")
        != {
            "every_controller_rounds": 1,
            "keep_history_tail": 1,
            "compact_only": True,
            "usage": CHECKPOINT_USAGE,
            "resume_consumable": False,
        }
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("paper_adoption_authorized") is not False
        or job.get("paper_evidence_adoption_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise PackageContractError("Sealed matched singleton job drifted.")

    matches = [
        row
        for row in _sequence(manifest.get("protocols"), label="protocol bindings")
        if isinstance(row, Mapping) and row.get("execution_id") == execution
    ]
    if len(matches) != 1:
        raise PackageContractError("Job protocol binding is not unique.")
    protocol_path, protocol_payload = _verify_binding(
        matches[0], label="protocol", canonical=True
    )
    assert protocol_payload is not None
    if (
        protocol_path != _package_path(job.get("protocol_path"), label="protocol path")
        or matches[0].get("canonical_sha256") != job.get("protocol_sha256")
        or matches[0].get("sha256") != job.get("protocol_file_sha256")
    ):
        raise PackageContractError("Job-to-protocol binding drifted.")
    _bundle_path, bundle = _verify_binding(
        manifest.get("bundle_manifest"), label="bundle manifest", canonical=True
    )
    _expected_path, expected = _verify_binding(
        manifest.get("bundle_expected_artifacts"),
        label="expected artifacts",
        canonical=True,
    )
    _locks_path, locks = _verify_binding(
        manifest.get("bundle_source_locks"), label="source locks", canonical=True
    )
    assert bundle is not None and expected is not None and locks is not None
    expected_cell = _mapping(
        _mapping(expected.get("cells"), label="expected cells").get(execution),
        label="expected cell",
    )
    if (
        bundle.get("cell_count") != 12
        or bundle.get("source_locks_sha256") != locks.get("sha256")
        or expected_cell.get("expected_run_artifacts")
        != job.get("expected_run_artifacts")
        or locks.get("sha256") != PARENT_SOURCE_LOCKS_CANONICAL_SHA256
        or locks.get("implementation_sources", {}).get("sha256")
        != PARENT_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
    ):
        raise PackageContractError("Matched bundle closure drifted.")
    return job, manifest, protocol_payload, locks


def _extract_source(manifest: Mapping[str, Any], destination: Path) -> None:
    archive_path, _ = _verify_binding(manifest.get("source_archive"), label="source archive")
    _source_path, source_manifest = _verify_binding(
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    assert source_manifest is not None
    if (
        source_manifest.get("archive") != manifest.get("source_archive")
        or source_manifest.get("status") != "passed"
        or source_manifest.get("archive_construction_no_ambient_repo_imports")
        is not True
        or source_manifest.get("execution_source_policy")
        != EXECUTION_SOURCE_POLICY
        or source_manifest.get("post_extraction_overlay_count") != 1
        or source_manifest.get("sealed_resume_reader")
        != {
            "path": SEALED_RESUME_READER.as_posix(),
            "sha256": SEALED_RESUME_READER_SHA256,
            "size_bytes": SEALED_RESUME_READER_SIZE_BYTES,
            "ambient_resume_overlay": False,
        }
        or source_manifest.get("parent_source_archive_sha256") != PARENT_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("additive_dependency")
        != {
            "path": APPEND_RUNTIME_DEPENDENCY.as_posix(),
            "sha256": APPEND_RUNTIME_DEPENDENCY_SHA256,
            "size_bytes": APPEND_RUNTIME_DEPENDENCY_SIZE_BYTES,
        }
    ):
        raise PackageContractError("Derived source archive authority drifted.")
    rows = _sequence(source_manifest.get("members"), label="source members")
    members = {
        safe_relative_path(row.get("path"), label="source member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(members) != len(rows) or len(rows) != int(source_manifest.get("member_count", -1)):
        raise PackageContractError("Derived source member closure drifted.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(member.name, label="tar member").as_posix()
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
            target = destination / relative
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
        raise PackageContractError("Derived source extraction is incomplete.")


def _apply_checkpoint_overlay(source_root: Path) -> dict[str, Any]:
    parity = load_json(STRONG5_PARITY_RECEIPT, label="checkpoint parity receipt")
    if (
        sha256_file(STRONG5_PARITY_RECEIPT) != STRONG5_PARITY_RECEIPT_FILE_SHA256
        or verify_self_digest(parity, label="checkpoint parity receipt")
        != STRONG5_PARITY_RECEIPT_CANONICAL_SHA256
        or parity.get("status") != "passed_exact_scientific_parity"
    ):
        raise PackageContractError("Checkpoint overlay parity authority drifted.")
    sealed = source_root / OPERATIONAL_CHECKPOINT_OVERLAY
    candidate = REPO_ROOT / OPERATIONAL_CHECKPOINT_OVERLAY
    sealed_resume = source_root / SEALED_RESUME_READER
    if (
        sealed_resume.is_symlink()
        or not sealed_resume.is_file()
        or sealed_resume.stat().st_size != SEALED_RESUME_READER_SIZE_BYTES
        or sha256_file(sealed_resume) != SEALED_RESUME_READER_SHA256
    ):
        raise PackageContractError("Sealed checkpoint resume reader drifted.")
    if sha256_file(sealed) != SEALED_CHECKPOINT_SHA256:
        raise PackageContractError("Sealed checkpoint implementation drifted.")
    if sha256_file(candidate) != OPERATIONAL_CHECKPOINT_OVERLAY_SHA256:
        raise PackageContractError("Candidate checkpoint implementation drifted.")
    shutil.copyfile(candidate, sealed)
    if sha256_file(sealed) != OPERATIONAL_CHECKPOINT_OVERLAY_SHA256:
        raise PackageContractError("Checkpoint overlay copy drifted.")
    if sha256_file(sealed_resume) != SEALED_RESUME_READER_SHA256:
        raise PackageContractError("Checkpoint overlay modified the resume reader.")
    return {
        "relative_path": OPERATIONAL_CHECKPOINT_OVERLAY.as_posix(),
        "sealed_sha256": SEALED_CHECKPOINT_SHA256,
        "candidate_sha256": OPERATIONAL_CHECKPOINT_OVERLAY_SHA256,
        "execution_source_policy": EXECUTION_SOURCE_POLICY,
        "post_extraction_overlay_count": 1,
        "ambient_resume_overlay": False,
        "sealed_resume_reader_path": SEALED_RESUME_READER.as_posix(),
        "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
        "fresh_start_only": True,
        "checkpoint_usage": CHECKPOINT_USAGE,
        "checkpoint_resume_authorized": False,
        "parity_canary_scope": PARITY_CANARY_SCOPE,
        "multi_round_compact_tail_resume_validated": False,
        "parity_receipt_sha256": parity["sha256"],
    }


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if name == "pipelines" or name.startswith("pipelines.") or name == "src" or name.startswith("src."):
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
    module = importlib.import_module("pipelines.static_adapt.ra_adapt")
    try:
        Path(str(module.__file__)).resolve().relative_to(root)
    except ValueError as exc:
        raise PackageContractError("Runtime implementation escaped the source archive.") from exc


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import resolve_problem_context

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
            n_fermions=None if receipt.n_fermions is None else int(receipt.n_fermions),
        )
    )


def _load_protocol(
    *, job: Mapping[str, Any], payload: Mapping[str, Any], source_locks: Mapping[str, Any]
) -> tuple[Any, Any]:
    from pipelines.static_adapt.ra_adapt.bundles import BundleCellSpec, _source_lock_refs
    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    receipt = protocol.bundle_materialization
    method = str(job["method"])
    expected_bundle = PARENT_BUNDLE_ID if method == "ra_singleton_plateau" else BUNDLE_ID
    expected_manifest_sha = (
        PARENT_BUNDLE_MANIFEST_CANONICAL_SHA256
        if method == "ra_singleton_plateau"
        else str(job["protocol_bundle_manifest_sha256"])
    )
    expected_algorithm = RA_ALGORITHM_ID if method == "ra_singleton_plateau" else APPEND_ALGORITHM_ID
    expected_adapter = RA_ADAPTER_ID if method == "ra_singleton_plateau" else APPEND_ADAPTER_ID
    if (
        receipt is None
        or receipt.bundle_id != expected_bundle
        or receipt.bundle_manifest_sha256 != expected_manifest_sha
        or receipt.source_locks_sha256 != PARENT_SOURCE_LOCKS_CANONICAL_SHA256
        or receipt.cell_id != job["execution_id"]
        or protocol.sha256 != job["protocol_sha256"]
        or protocol.algorithm_id != expected_algorithm
        or protocol.adapter_id != expected_adapter
        or protocol.horizon != TARGET_HORIZON
        or protocol.optimizer != "powell"
        or protocol.optimizer_maxiter != 200
        or protocol.seeds != {"adapt": 7, "transpiler": 7}
        or protocol.active_gradient_policy != ACTIVE_GRADIENT_POLICY
        or protocol.resource_weighting_scope != RESOURCE_WEIGHTING_SCOPE
        or protocol.candidate_representation != CANDIDATE_REPRESENTATION
        or protocol.problem.num_sites != 2
        or protocol.problem.n_ph_max != int(job["nph"])
        or protocol.request.execution.stop.maximum_controller_rounds != TARGET_HORIZON
        or protocol.request.execution.resume.kind != "fresh_start"
    ):
        raise PackageContractError("Typed matched singleton protocol drifted.")
    if method == "ra_singleton_plateau":
        route = protocol.route_contract
        method_invalid = (
            route.get("sha256") != RA_ROUTE_CONTRACT_SHA256
            or protocol.request.method.admission.kind != "singleton"
            or protocol.request.method.insertion.kind != "plateau_commutation"
        )
    else:
        method_invalid = (
            protocol.schema != "paper_i_append_adapt_resolved_protocol_v1"
            or protocol.selector_identity != APPEND_SELECTOR_ID
            or protocol.selector_scope != APPEND_SELECTOR_SCOPE
            or protocol.lineage_authority.get("ra_staged_funnel_invoked") is not False
            or protocol.request.observation.checkpoint.keep_history_tail != 1
            or hasattr(protocol.request, "method")
        )
    if method_invalid:
        raise PackageContractError("Method-specific protocol contract drifted.")
    cell = BundleCellSpec(
        cell_id=str(job["execution_id"]),
        stage=f"page12_matched_{job['regime_id']}_{method}_candidate",
        regime_id=str(job["regime_id"]),
        nph=int(job["nph"]),
        route_id=str(job["route_id"]),
        algorithm_id=str(job["algorithm_id"]),
        selector_family="ra_adapt" if method == "ra_singleton_plateau" else "append_adapt",
        candidate_representation=CANDIDATE_REPRESENTATION,
        horizon=TARGET_HORIZON,
        source_lock_id=source_lock_id(str(job["regime_id"]), int(job["nph"])),
    )
    refs = _source_lock_refs(source_locks, cell=cell)
    authority = _mint_bundle_protocol_materialization_authority(
        receipt, source_lock_refs=refs, protocol_sha256=protocol.sha256
    )
    protocol = _attach_validated_bundle_protocol_authority(protocol, authority)
    return protocol, _problem_from_protocol(protocol)


def _prepare(job_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any, Any, dict[str, Any], tempfile.TemporaryDirectory[str]]:
    job, manifest, payload, locks = _load_closed_job(job_path)
    temporary = tempfile.TemporaryDirectory(prefix=f"paper-i-page12-matched12-{job['execution_id']}.")
    try:
        source_root = Path(temporary.name) / "source"
        _extract_source(manifest, source_root)
        overlay = _apply_checkpoint_overlay(source_root)
        original = Path.cwd()
        os.chdir(source_root)
        try:
            _activate_source_root(source_root)
            protocol, problem = _load_protocol(job=job, payload=payload, source_locks=locks)
        finally:
            os.chdir(original)
    except BaseException:
        temporary.cleanup()
        raise
    return job, manifest, protocol, problem, overlay, temporary


def preflight(job_path: Path) -> dict[str, Any]:
    job, manifest, protocol, _problem, overlay, temporary = _prepare(job_path)
    temporary.cleanup()
    return digested(
        {
            "schema": WORKER_PREFLIGHT_SCHEMA,
            "status": "passed",
            "execution_id": job["execution_id"],
            "method": job["method"],
            "job_spec_sha256": job["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "protocol_sha256": protocol.sha256,
            "route_contract_sha256": job["route_contract_sha256"],
            "source_archive_import_isolated": True,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "checkpoint_overlay": overlay,
            "compact_checkpoint_keep_history_tail": 1,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "fresh_start": True,
            "fresh_start_only": True,
            "checkpoint_resume_authorized": False,
            "target_horizon": TARGET_HORIZON,
            "scientific_execution_performed": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _validate_activation(path: Path, *, job: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    activation = load_json(path, label="local activation")
    verify_self_digest(activation, label="local activation")
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status") != "authorized_local_execution"
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("execution_authorized") is not True
        or activation.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or activation.get("fresh_start_only") is not True
        or activation.get("checkpoint_usage") != CHECKPOINT_USAGE
        or activation.get("checkpoint_resume_authorized") is not False
        or activation.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or activation.get("submission_authorized") is not False
        or activation.get("paper_adoption_authorized") is not False
        or activation.get("paper_evidence_adoption_authorized") is not False
        or job.get("execution_id") not in activation.get("execution_ids", [])
    ):
        raise PackageContractError("Local execution activation drifted.")
    return activation


def _validate_absolute_binding(
    raw: Any, *, expected_path: Path, label: str, canonical: bool
) -> dict[str, Any] | None:
    binding = _mapping(raw, label=f"{label} binding")
    path = Path(str(binding.get("path", "")))
    if (
        not path.is_absolute()
        or path.resolve() != expected_path.resolve()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
        or sha256_file(path) != binding.get("file_sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != binding.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return payload


def _validate_child_capability(
    *,
    activation_path: Path,
    output_dir: Path,
    receipt_path: Path,
    execution_id: str,
    child_token: str,
) -> dict[str, Any]:
    expected_output = DEFAULT_RUNTIME_DIR / "runs" / execution_id
    expected_receipt = DEFAULT_RUNTIME_DIR / "receipts" / f"{execution_id}.json"
    if (
        activation_path.resolve() != DEFAULT_ACTIVATION_PATH.resolve()
        or output_dir.resolve() != expected_output.resolve()
        or receipt_path.resolve() != expected_receipt.resolve()
    ):
        raise PackageContractError("Worker paths escaped the pinned child seam.")
    activation = load_json(activation_path, label="pinned child activation")
    verify_self_digest(activation, label="pinned child activation")
    receipt_env = os.environ.get(HANDOFF_RECEIPT_ENV)
    if (
        receipt_env is None
        or Path(receipt_env).resolve() != DEFAULT_HANDOFF_RECEIPT.resolve()
    ):
        raise PackageContractError("Pinned handoff receipt environment drifted.")
    handoff = load_json(DEFAULT_HANDOFF_RECEIPT, label="pinned handoff receipt")
    verify_self_digest(handoff, label="pinned handoff receipt")
    launch_token = hashlib.sha256(
        f"{handoff['sha256']}:matched-singleton12-target-launch-v1".encode(
            "utf-8"
        )
    ).hexdigest()
    expected_child_token = hashlib.sha256(
        (
            f"{activation['sha256']}:{handoff['sha256']}:{execution_id}:"
            "matched12-child-v1"
        ).encode("utf-8")
    ).hexdigest()
    if (
        handoff.get("schema") != HANDOFF_RECEIPT_SCHEMA
        or handoff.get("status") != HANDOFF_RECEIPT_STATUS
        or handoff.get("target_activation_manifest_sha256")
        != activation.get("sha256")
        or handoff.get("target_command")
        != [
            "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
            "-B",
            RUNNER_PATH.as_posix(),
            "--run-campaign",
        ]
        or handoff.get("target_environment") != REQUIRED_NUMERICAL_ENVIRONMENT
        or handoff.get("execution_authorized") is not True
        or handoff.get("submission_authorized") is not False
        or handoff.get("paper_adoption_authorized") is not False
        or handoff.get("paper_evidence_adoption_authorized") is not False
        or os.environ.get(HANDOFF_TOKEN_ENV) != launch_token
        or os.environ.get(CHILD_TOKEN_ENV) != expected_child_token
        or child_token != expected_child_token
    ):
        raise PackageContractError("Runner child capability drifted.")
    target_contract = _validate_absolute_binding(
        handoff.get("target_contract"),
        expected_path=DEFAULT_TARGET_CONTRACT,
        label="target contract",
        canonical=True,
    )
    activation_binding = _mapping(
        handoff.get("target_authority_bindings"),
        label="target authority bindings",
    ).get("activation_manifest")
    bound_activation = _validate_absolute_binding(
        activation_binding,
        expected_path=DEFAULT_ACTIVATION_PATH,
        label="target activation",
        canonical=True,
    )
    if (
        target_contract is None
        or target_contract.get("target", {}).get("activation_dir")
        != DEFAULT_ACTIVATION_PATH.parent.resolve().as_posix()
        or bound_activation != activation
    ):
        raise PackageContractError("Child capability authority binding drifted.")
    try:
        descriptor = int(os.environ[HANDOFF_LOCK_FD_ENV])
        inherited = os.fstat(descriptor)
        expected_lock = DEFAULT_HANDOFF_LOCK.lstat()
        if (
            DEFAULT_HANDOFF_LOCK.is_symlink()
            or not stat.S_ISREG(expected_lock.st_mode)
            or inherited.st_dev != expected_lock.st_dev
            or inherited.st_ino != expected_lock.st_ino
        ):
            raise PackageContractError("Inherited handoff lock inode drifted.")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (KeyError, OSError, ValueError) as exc:
        raise PackageContractError("Inherited handoff lock proof failed.") from exc
    try:
        command = psutil.Process().cmdline()
    except psutil.Error as exc:
        raise PackageContractError("Worker process command is unavailable.") from exc
    if command[-4:] != [
        "-B",
        RUNNER_PATH.as_posix(),
        "--child-cell",
        execution_id,
    ]:
        raise PackageContractError("Worker was not invoked by the runner child seam.")
    if {
        key: os.environ.get(key) for key in REQUIRED_NUMERICAL_ENVIRONMENT
    } != REQUIRED_NUMERICAL_ENVIRONMENT:
        raise PackageContractError("Worker numerical environment drifted.")
    return activation


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _configuration_snapshot(module: Any) -> Any:
    configuration = getattr(module, "__config__", None)
    show = getattr(configuration, "show", None)
    if not callable(show):
        return {"available": False}
    try:
        return _jsonable(show(mode="dicts"))
    except TypeError:
        import contextlib
        import io

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            show()
        return {"available": True, "text": output.getvalue()}


def _sysctl(name: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error_type": type(exc).__name__}
    return {
        "available": completed.returncode == 0,
        "value": completed.stdout.strip() if completed.returncode == 0 else None,
        "returncode": int(completed.returncode),
    }


def _mac_hardware_identity() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {
            "available": False,
            "source": "system_profiler_SPHardwareDataType_allowlist_v1",
            "chip_type": None,
            "machine_model": None,
            "machine_name": None,
            "number_processors": None,
        }
    try:
        completed = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        payload = {}
    rows = payload.get("SPHardwareDataType") if isinstance(payload, Mapping) else None
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) else {}
    selected = {
        "chip_type": row.get("chip_type"),
        "machine_model": row.get("machine_model"),
        "machine_name": row.get("machine_name"),
        "number_processors": row.get("number_processors"),
    }
    return {
        "available": bool(selected["chip_type"] and selected["machine_model"]),
        "source": "system_profiler_SPHardwareDataType_allowlist_v1",
        **selected,
    }


def _loaded_dyld_images(*, markers: tuple[str, ...]) -> list[str]:
    library = ctypes.CDLL(None)
    image_count = getattr(library, "_dyld_image_count")
    image_name = getattr(library, "_dyld_get_image_name")
    image_count.restype = ctypes.c_uint32
    image_name.argtypes = [ctypes.c_uint32]
    image_name.restype = ctypes.c_char_p
    paths: set[str] = set()
    for index in range(int(image_count())):
        raw = image_name(index)
        if raw is None:
            continue
        path = raw.decode("utf-8", errors="strict")
        lowered = path.lower()
        if any(marker in lowered for marker in markers):
            paths.add(path)
    return sorted(paths)


def _native_runtime_receipt(job: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the loaded native numerical runtime immediately before a cell."""

    import importlib.metadata
    import numpy as np
    import scipy
    import scipy.linalg
    import qiskit

    # Force NumPy/SciPy BLAS and LAPACK dispatch before inspecting loaded
    # libraries.  These tiny identity operations are runtime probes only.
    matrix = np.eye(2, dtype=float)
    np.linalg.svd(matrix)
    scipy.linalg.blas.dgemm(alpha=1.0, a=matrix, b=matrix)
    try:
        import qiskit_aer  # noqa: F401
    except ImportError:
        pass

    versions: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "qiskit",
        "qiskit-aer",
        "psutil",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        affinity: dict[str, Any] = {
            "available": True,
            "cpu_indices": list(psutil.Process().cpu_affinity()),
        }
    except (AttributeError, NotImplementedError, psutil.Error):
        affinity = {"available": False, "cpu_indices": None}
    try:
        multiarray = importlib.import_module("numpy._core._multiarray_umath")
        cpu_features = _jsonable(getattr(multiarray, "__cpu_features__", {}))
    except ImportError:
        cpu_features = {}
    loaded_libraries = _loaded_dyld_images(
        markers=(
            "accelerate.framework",
            "libblas",
            "liblapack",
            "openblas",
            "libmkl",
            "_fblas",
            "_flapack",
            "cython_blas",
            "cython_lapack",
        )
    )
    libc_images = _loaded_dyld_images(
        markers=(
            "libsystem.b.dylib",
            "/usr/lib/system/libsystem",
            "/libc.",
            "libc.so",
        )
    )
    libc_name, libc_version = platform.libc_ver()
    darwin_libsystem_version = (
        platform.mac_ver()[0] or platform.release()
        if platform.system() == "Darwin"
        else None
    )
    libc_identity = {
        "platform_libc_ver": {
            "available": bool(libc_name or libc_version),
            "name": libc_name or None,
            "version": libc_version or None,
        },
        "loaded_image_evidence_available": bool(libc_images),
        "loaded_images": [
            {
                "path": path,
                "version": darwin_libsystem_version,
                "version_source": "platform_mac_ver_for_darwin_libsystem_v1",
            }
            for path in libc_images
        ],
        "darwin_libsystem_version": darwin_libsystem_version,
    }
    accelerate_paths = [
        path for path in loaded_libraries if "accelerate.framework" in path.lower()
    ]
    threadpools = []
    if accelerate_paths:
        threadpools.append(
            {
                "user_api": "blas",
                "internal_api": "accelerate",
                "prefix": "accelerate",
                "filepath": accelerate_paths[0],
                "version": platform.mac_ver()[0] or platform.release(),
                "num_threads": int(os.environ.get("VECLIB_MAXIMUM_THREADS", "0")),
                "thread_count_source": "VECLIB_MAXIMUM_THREADS_process_environment_v1",
                "identity_source": "loaded_dyld_image_plus_numpy_scipy_build_contract_v1",
            }
        )
    executable = Path(sys.executable).resolve()
    payload = {
        "schema": NATIVE_RUNTIME_SCHEMA,
        "observed_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "execution_id": str(job["execution_id"]),
        "method": str(job["method"]),
        "python": {
            "executable": Path(sys.executable).as_posix(),
            "executable_resolved": executable.as_posix(),
            "executable_sha256": sha256_file(executable),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": list(platform.uname()),
        },
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "brand_string": _sysctl("machdep.cpu.brand_string"),
            "features": _sysctl("machdep.cpu.features"),
            "leaf7_features": _sysctl("machdep.cpu.leaf7_features"),
            "numpy_dispatch_features": cpu_features,
            "mac_hardware_identity": _mac_hardware_identity(),
            "affinity": affinity,
        },
        "packages": versions,
        "numpy_configuration": _configuration_snapshot(np),
        "scipy_configuration": _configuration_snapshot(scipy),
        "loaded_threadpools": threadpools,
        "loaded_blas_lapack_libraries": loaded_libraries,
        "libc_identity": libc_identity,
        "threadpoolctl_available": False,
        "resource_contract": {
            "kind": "native_local_cpu_only_serial_v1",
            "job_requested_cpu_count": int(job["resources"]["request_cpus"]),
            "scheduler_allocation_available": False,
            "scheduler_allocated_cpu_count": None,
            "native_local_host_logical_cpu_count": psutil.cpu_count(logical=True),
            "process_affinity_available": affinity["available"],
            "process_affinity_cpu_count": (
                len(affinity["cpu_indices"]) if affinity["available"] else None
            ),
            "numerical_kernel_thread_count": 1,
            "maximum_campaign_concurrency": 1,
            "gpu_requested_count": 0,
            "gpu_execution_authorized": False,
            "gpu_execution_active": False,
        },
        "numerical_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "OMP_DYNAMIC",
                "MKL_DYNAMIC",
                "STATIC_ADAPT_HH_POOL_CACHE",
                "STATIC_ADAPT_CANDIDATE_RECORD_CACHE",
            )
        },
        "capture_point": "inside_cell_after_numpy_scipy_qiskit_blas_load_before_scientific_execution_v1",
        "scientific_execution_performed": False,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    return digested(payload)


def _execute(method: str, protocol: Any, problem: Any, staging: Path) -> tuple[Any, int]:
    from pipelines.static_adapt.ra_adapt import RAAdaptOperationalControls, run_ra_adapt
    from pipelines.static_adapt.sr_snake import (
        CheckpointObservation,
        EstimatorLedgerObservation,
        FreshStart,
        SRObservationPolicy,
    )

    (staging / "checkpoints").mkdir(parents=True, exist_ok=False)
    (staging / "result").mkdir(parents=True, exist_ok=False)
    (staging / "summary").mkdir(parents=True, exist_ok=False)
    controls = RAAdaptOperationalControls(
        maximum_controller_rounds=TARGET_HORIZON,
        resume=FreshStart(),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=staging / "checkpoints/current.json",
                every_controller_rounds=1,
                keep_history_tail=1,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=staging / "result/estimator_ledger.json"
            ),
            resource_rounds=(TARGET_HORIZON,),
        ),
    )
    if method == "ra_singleton_plateau":
        result = run_ra_adapt(problem, protocol, operational_controls=controls)
        rounds = len(result.run.accepted_trajectory)
    elif method == "append_singleton":
        from pipelines.static_adapt.ra_adapt import run_append_adapt

        original = Path.cwd()
        os.chdir(staging)
        try:
            result = run_append_adapt(problem, protocol)
        finally:
            os.chdir(original)
        rounds = int(result.result_payload["controller_rounds_completed"])
    else:
        raise PackageContractError("Unknown execution facade.")
    if (
        result.protocol.sha256 != protocol.sha256
        or not 1 <= rounds <= TARGET_HORIZON
        or not (staging / "checkpoints/current.json").is_file()
        or not (staging / "result/estimator_ledger.json").is_file()
    ):
        raise PackageContractError("Scientific execution closure failed.")
    return result, rounds


def run_cell(
    *,
    job_path: Path,
    activation_path: Path,
    output_dir: Path,
    receipt_path: Path,
    child_token: str,
) -> dict[str, Any]:
    capability_activation = _validate_child_capability(
        activation_path=activation_path,
        output_dir=output_dir,
        receipt_path=receipt_path,
        execution_id=output_dir.name,
        child_token=child_token,
    )
    job, manifest, protocol, problem, overlay, temporary = _prepare(job_path)
    try:
        activation = _validate_activation(activation_path, job=job, manifest=manifest)
        if (
            activation != capability_activation
            or output_dir.name != job["execution_id"]
            or output_dir.exists()
            or output_dir.is_symlink()
            or receipt_path.exists()
            or receipt_path.is_symlink()
            or output_dir.parent.name != "runs"
        ):
            raise PackageContractError("Worker destination is not a fresh cell path.")
        source_root = Path(temporary.name) / "source"
        staging = Path(temporary.name) / "cell_output"
        staging.mkdir()
        (staging / "runtime").mkdir()
        native_runtime = _native_runtime_receipt(job)
        _write_json(staging / "runtime/native_runtime.json", native_runtime)
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = _execute(str(job["method"]), protocol, problem, staging)
        finally:
            os.chdir(original)
        _write_json(staging / "result/result.json", result.to_dict())
        summary = result.run.paper_i_summary if job["method"] == "ra_singleton_plateau" else result.paper_i_summary
        if summary is None:
            raise PackageContractError("Paper-I summary is required.")
        _write_json(staging / "summary/summary.json", summary.to_dict())
        preliminary: dict[str, Any] = {}
        for role, row in job["expected_run_artifacts"].items():
            if role == "execution_manifest":
                continue
            relative = PurePosixPath(str(row["path"])).relative_to(
                PurePosixPath("runs") / str(job["execution_id"])
            )
            path = staging / Path(*relative.parts)
            if not path.is_file():
                raise PackageContractError(f"Required cell artifact is absent: {role}")
            preliminary[role] = {
                "path": str(row["path"]),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        execution_manifest = digested(
            {
                "schema": EXECUTION_MANIFEST_SCHEMA,
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "method": job["method"],
                "job_spec_sha256": job["sha256"],
                "activation_sha256": activation["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": job["route_contract_sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "fresh_start_only": True,
                "compact_checkpoint_keep_history_tail": 1,
                "checkpoint_usage": CHECKPOINT_USAGE,
                "checkpoint_resume_authorized": False,
                "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                "checkpoint_overlay": overlay,
                "native_runtime_receipt": {
                    "path": (
                        f"runs/{job['execution_id']}/runtime/native_runtime.json"
                    ),
                    "sha256": native_runtime["sha256"],
                    "file_sha256": sha256_file(
                        staging / "runtime/native_runtime.json"
                    ),
                    "size_bytes": (
                        staging / "runtime/native_runtime.json"
                    ).stat().st_size,
                },
                "output_payloads": preliminary,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json(staging / "execution_manifest.json", execution_manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, output_dir)
        receipt = digested(
            {
                "schema": WORKER_RECEIPT_SCHEMA,
                "status": "passed",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "method": job["method"],
                "job_spec_sha256": job["sha256"],
                "activation_sha256": activation["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "fresh_start_only": True,
                "compact_checkpoint_keep_history_tail": 1,
                "checkpoint_usage": CHECKPOINT_USAGE,
                "checkpoint_resume_authorized": False,
                "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                "native_runtime_receipt": native_runtime,
                "native_runtime_receipt_file_sha256": sha256_file(
                    output_dir / "runtime/native_runtime.json"
                ),
                "artifacts": [
                    {
                        "path": path.relative_to(output_dir.parent.parent).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(output_dir.rglob("*"))
                    if path.is_file()
                ],
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json(receipt_path, receipt)
        return receipt
    finally:
        temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", required=True)
    args = parser.parse_args()
    try:
        payload = preflight(args.job)
        print(canonical_json_bytes(payload).decode("utf-8"))
        return 0
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
