#!/usr/bin/env python3
"""Build the immutable, authorization-bound Paper-I Study-1 CHTC package.

This builder never invents authorization and never submits work.  Its normal
mode requires the passed v8 materialization receipt, the frozen P2/P3
scientific-preflight receipt, and a separately supplied execution/submission
authorization receipt. ``--validate-only`` performs the same authority checks
without writing package artifacts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    BUNDLE_IDS,
    CAMPAIGN_ID,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    EXPECTED_ARTIFACT_ROLES,
    JOB_SPEC_SCHEMA,
    MACOS_SF_DATALESS,
    MACOS_UF_COMPRESSED,
    MAX_RUNTIME_SECONDS,
    PACKAGE_CONTROL_PLANE_FILES,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUEST_CPUS,
    RUN_CLASS,
    SCIENTIFIC_PREFLIGHT_RELATIVE,
    V8_REVISION,
    PackageContractError,
    atomic_write_json,
    digested,
    direct_execution_rows,
    expected_artifact_path,
    load_json_object,
    logical_cell_keys,
    objective_gate_diagnostic_contract,
    package_control_plane_receipt,
    repo_root_from_script,
    resource_envelope,
    safe_relative_path,
    sha256_file,
    shared_append_rows,
    stage_packaged_runtime_tree,
    submission_preflight_overlay_contract,
    validate_authorization_receipt,
    validate_scientific_preflight_receipt,
    validate_v8_authority,
    validation_cell_id,
)


STATIC_PACKAGE_FILES = PACKAGE_CONTROL_PLANE_FILES
GENERATED_TOP_LEVEL_FILES = (
    "execution_plan.json",
    "queue.tsv",
    "source_archive_manifest.json",
)
V8_FINAL_AUTHORITY_COPY = "authority/v8_final_materialization_receipt.json"
OBJECTIVE_GATE_AUTHORITY_COPY = (
    "authority/study1_objective_gate_authority_receipt.json"
)
EXECUTION_AUTHORITY_COPY = "authority/execution_authorization_receipt.json"
AUTHORITY_COPY_FILES = (
    V8_FINAL_AUTHORITY_COPY,
    OBJECTIVE_GATE_AUTHORITY_COPY,
    EXECUTION_AUTHORITY_COPY,
    SCIENTIFIC_PREFLIGHT_RELATIVE,
)
MUTABLE_RUNTIME_DIRS = ("fetched", "logs")


def _exclusive_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PackageContractError(f"Refusing to overwrite package file: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise PackageContractError(
            f"Refusing to overwrite stale package temporary: {temporary}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_exact(source: Path, destination: Path) -> None:
    _exclusive_write_bytes(destination, source.read_bytes())


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(
            f"Package input escapes the active repository: {path}"
        ) from exc


def _collect_archive_members(
    *,
    repo_root: Path,
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for row in authority["verified_source_files"]:
        relative = safe_relative_path(
            row["path"], label="verified source archive path"
        ).as_posix()
        members[relative] = {
            "path": relative,
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
            "source_kind": "verified_implementation_inventory",
        }

    trusted = authority["objective_gate_authority"]["trusted"]
    for member_list_name in (
        "controller_instrumentation_members",
        "reporting_boundary_members",
    ):
        raw_members = trusted.get(member_list_name)
        if not isinstance(raw_members, list) or not raw_members:
            raise PackageContractError(
                f"G8 trusted authority has no {member_list_name}."
            )
        for raw in raw_members:
            if not isinstance(raw, Mapping):
                raise PackageContractError(
                    f"G8 {member_list_name} contains an invalid member."
                )
            relative = safe_relative_path(
                raw.get("path"),
                label=f"G8 {member_list_name} source path",
            ).as_posix()
            source = repo_root / relative
            if not source.is_file() or source.is_symlink():
                raise PackageContractError(
                    f"G8 trusted source member is unavailable: {relative}"
                )
            source_flags = int(getattr(source.stat(), "st_flags", 0))
            if source_flags & (MACOS_UF_COMPRESSED | MACOS_SF_DATALESS):
                raise PackageContractError(
                    "G8 trusted source member is compressed/dataless: "
                    f"{relative}"
                )
            expected_sha256 = str(raw.get("sha256", ""))
            actual_sha256 = sha256_file(source)
            if actual_sha256 != expected_sha256:
                raise PackageContractError(
                    f"G8 trusted source member drifted: {relative}: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            binding = {
                "path": relative,
                "sha256": expected_sha256,
                "size_bytes": source.stat().st_size,
                "source_kind": "g8_trusted_source_dataflow_member",
            }
            previous = members.get(relative)
            if previous is not None:
                if (
                    previous.get("sha256") != binding["sha256"]
                    or previous.get("size_bytes") != binding["size_bytes"]
                ):
                    raise PackageContractError(
                        f"Source archive member collision: {relative}"
                    )
                continue
            members[relative] = binding

    revision_root = Path(authority["v8_root"])
    for bundle_id in BUNDLE_IDS:
        bundle_dir = revision_root / bundle_id
        for candidate in sorted(bundle_dir.rglob("*")):
            relative_inside = candidate.relative_to(bundle_dir)
            if candidate.is_symlink():
                raise PackageContractError(
                    f"v8 bundle contains a forbidden symlink: {candidate}"
                )
            if any(
                part in {"runs", "__pycache__"} or part.startswith(".")
                for part in relative_inside.parts
            ):
                raise PackageContractError(
                    f"v8 bundle contains a mutable/hidden path: {candidate}"
                )
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise PackageContractError(
                    f"v8 bundle contains a non-regular member: {candidate}"
                )
            relative = _repo_relative(candidate, repo_root)
            binding = {
                "path": relative,
                "sha256": sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
                "source_kind": "immutable_v8_bundle",
                "bundle_id": bundle_id,
            }
            previous = members.get(relative)
            if previous is not None and previous != binding:
                raise PackageContractError(
                    f"Source archive member collision: {relative}"
                )
            members[relative] = binding

    ordered = [members[key] for key in sorted(members)]
    if not ordered:
        raise PackageContractError("Source-locked archive would be empty.")
    return ordered


def _build_deterministic_archive(
    *,
    repo_root: Path,
    destination: Path,
    members: Iterable[Mapping[str, Any]],
) -> None:
    if destination.exists():
        raise PackageContractError(
            f"Refusing to overwrite source archive: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise PackageContractError(
            f"Refusing to overwrite stale source archive: {temporary}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for row in members:
                        relative = safe_relative_path(
                            row["path"], label="source archive member"
                        ).as_posix()
                        source = repo_root / relative
                        if (
                            not source.is_file()
                            or source.is_symlink()
                            or sha256_file(source) != row["sha256"]
                            or source.stat().st_size != int(row["size_bytes"])
                        ):
                            raise PackageContractError(
                                f"Source archive member drifted: {relative}"
                            )
                        info = tarfile.TarInfo(relative)
                        info.size = source.stat().st_size
                        info.mode = (
                            0o755
                            if source.stat().st_mode & 0o111
                            else 0o644
                        )
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_expected_cell(
    *,
    repo_root: Path,
    bundle_binding: Mapping[str, Any],
    cell_id: str,
) -> Mapping[str, Any]:
    expected = load_json_object(
        repo_root / bundle_binding["expected_artifacts"]["path"],
        label="v7 expected-artifact index",
    )
    cell = expected.get("cells", {}).get(cell_id)
    if not isinstance(cell, Mapping):
        raise PackageContractError(
            f"Expected-artifact cell is unavailable: {cell_id}"
        )
    return cell


def _logical_rows(
    *,
    repo_root: Path,
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    shared_by_reference = {
        row["reference_logical_key"]: row for row in shared_append_rows()
    }
    rows: list[dict[str, Any]] = []
    for bundle_id in BUNDLE_IDS:
        binding = authority["bundle_bindings"][bundle_id]
        for regime_id in ("strong_weak_u8", "strong_strong_u8"):
            for route_id in (
                "append_macro",
                "ra_macro_append_only",
                "ra_macro_plateau",
                "ra_macro_always",
                "singleton_plateau",
            ):
                cell_id = validation_cell_id(regime_id, route_id)
                logical_key = f"{bundle_id}::{cell_id}"
                expected_cell = _load_expected_cell(
                    repo_root=repo_root,
                    bundle_binding=binding,
                    cell_id=cell_id,
                )
                fulfillment = dict(expected_cell["execution_fulfillment"])
                reference = shared_by_reference.get(logical_key)
                rows.append(
                    {
                        "logical_key": logical_key,
                        "bundle_id": bundle_id,
                        "cell_id": cell_id,
                        "regime_id": regime_id,
                        "route_id": route_id,
                        "direct_execution_required": reference is None,
                        "canonical_execution_id": (
                            reference["canonical_execution_id"]
                            if reference is not None
                            else f"{bundle_id}__{cell_id}"
                        ),
                        "protocol": dict(
                            binding["validation_protocols"][cell_id]
                        ),
                        "execution_template": dict(
                            binding[
                                "validation_execution_templates"
                            ][cell_id]
                        ),
                        "execution_fulfillment": fulfillment,
                        "expected_run_artifacts": {
                            role: dict(
                                expected_cell["expected_run_artifacts"][role]
                            )
                            for role in EXPECTED_ARTIFACT_ROLES
                        },
                    }
                )
    if [row["logical_key"] for row in rows] != list(logical_cell_keys()):
        raise PackageContractError("Logical Study-1 ordering drifted.")
    if sum(bool(row["direct_execution_required"]) for row in rows) != 18:
        raise PackageContractError("Logical Study-1 direct count drifted.")
    return rows


def _build_plan(
    *,
    repo_root: Path,
    authority: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_file_sha256: str,
    source_manifest: Mapping[str, Any],
    source_archive_sha256: str,
    control_plane: Mapping[str, Any],
) -> dict[str, Any]:
    logical = _logical_rows(repo_root=repo_root, authority=authority)
    logical_by_key = {row["logical_key"]: row for row in logical}
    direct: list[dict[str, Any]] = []
    for identity in direct_execution_rows():
        logical_key = f"{identity['bundle_id']}::{identity['cell_id']}"
        logical_row = logical_by_key[logical_key]
        memory_mb, disk_mb = resource_envelope(
            identity["regime_id"], identity["route_id"]
        )
        direct.append(
            {
                **identity,
                "logical_key": logical_key,
                "execution_entrypoint": (
                    "run_append_adapt"
                    if identity["route_id"] == "append_macro"
                    else "run_ra_adapt"
                ),
                "protocol": dict(logical_row["protocol"]),
                "execution_template": dict(
                    logical_row["execution_template"]
                ),
                "execution_fulfillment": dict(
                    logical_row["execution_fulfillment"]
                ),
                "artifact_paths": {
                    role: (
                        f"{authority['bundle_bindings'][identity['bundle_id']]['bundle_root']}/"
                        f"{expected_artifact_path(identity['cell_id'], role)}"
                    )
                    for role in EXPECTED_ARTIFACT_ROLES
                },
                "objective_gate_diagnostics": objective_gate_diagnostic_contract(
                    bundle_id=identity["bundle_id"],
                    regime_id=identity["regime_id"],
                    route_id=identity["route_id"],
                ),
                "resources": {
                    "request_cpus": REQUEST_CPUS,
                    "request_memory_mb": memory_mb,
                    "request_disk_mb": disk_mb,
                    "max_runtime_seconds": MAX_RUNTIME_SECONDS,
                    "resource_envelope_basis": (
                        "completed_prior_paper_i_nph3_macro_and_"
                        "guarded_singleton_chtc_v1"
                    ),
                    "representation_class": (
                        "singleton"
                        if identity["route_id"] == "singleton_plateau"
                        else "macro"
                    ),
                },
            }
        )
    return digested(
        {
            "schema": EXECUTION_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "materialization_revision": V7_REVISION,
            "authorization": {
                "authorization_id": authorization["authorization_id"],
                "canonical_sha256": authorization["sha256"],
                "file_sha256": authorization_file_sha256,
            },
            "v7_final_receipt": dict(authority["final_receipt_binding"]),
            "study1_objective_gate_authority": dict(
                authority["objective_gate_authority_binding"]
            ),
            "study1_dedupe_sha256": authority["dedupe_sha256"],
            "package_control_plane": dict(control_plane),
            "source_inventory_sha256": authority["source_inventory"]["sha256"],
            "source_archive": {
                "path": "source_locked.tar.gz",
                "sha256": source_archive_sha256,
                "manifest_canonical_sha256": source_manifest["sha256"],
            },
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
            },
            "logical_cell_count": 20,
            "direct_execution_count": 18,
            "shared_reference_count": 2,
            "logical_cells": logical,
            "direct_executions": direct,
            "shared_append_references": [
                dict(row) for row in shared_append_rows()
            ],
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "not_submitted",
        }
    )


def _build_job_spec(
    *,
    plan: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return digested(
        {
            "schema": JOB_SPEC_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "execution_id": row["execution_id"],
            "bundle_id": row["bundle_id"],
            "cell_id": row["cell_id"],
            "regime_id": row["regime_id"],
            "route_id": row["route_id"],
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "execution_entrypoint": row["execution_entrypoint"],
            "execution_plan_sha256": plan["sha256"],
            "authorization": dict(plan["authorization"]),
            "v7_final_receipt": dict(plan["v7_final_receipt"]),
            "study1_objective_gate_authority": dict(
                plan["study1_objective_gate_authority"]
            ),
            "study1_dedupe_sha256": plan["study1_dedupe_sha256"],
            "package_control_plane": dict(plan["package_control_plane"]),
            "source_inventory_sha256": plan["source_inventory_sha256"],
            "source_archive": dict(plan["source_archive"]),
            "remote_image": dict(plan["remote_image"]),
            "protocol": dict(row["protocol"]),
            "execution_template": dict(row["execution_template"]),
            "execution_fulfillment": dict(row["execution_fulfillment"]),
            "artifact_paths": dict(row["artifact_paths"]),
            "objective_gate_diagnostics": dict(
                row["objective_gate_diagnostics"]
            ),
            "resources": dict(row["resources"]),
            "worker_receipt_path": (
                f"worker_receipts/{row['execution_id']}.json"
            ),
            "execution_authorized": True,
            "submission_authorized": True,
        }
    )


def _manifest_file_binding(package_dir: Path, relative: str) -> dict[str, Any]:
    path = package_dir / safe_relative_path(
        relative, label="package manifest file"
    )
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Missing package-only file: {path}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def _build_package_manifest(
    *,
    package_dir: Path,
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    authorization: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    job_relatives: list[str],
    control_plane: Mapping[str, Any],
) -> dict[str, Any]:
    relative_files = [
        *STATIC_PACKAGE_FILES,
        *GENERATED_TOP_LEVEL_FILES,
        *AUTHORITY_COPY_FILES,
        *job_relatives,
    ]
    if len(relative_files) != len(set(relative_files)):
        raise PackageContractError("Package-only manifest path list has duplicates.")
    return digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "materialization_revision": V7_REVISION,
            "execution_plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "v7_final_receipt_sha256": authority["final_receipt"]["sha256"],
            "study1_objective_gate_authority_sha256": authority[
                "objective_gate_authority"
            ]["sha256"],
            "package_control_plane_sha256": control_plane["sha256"],
            "source_archive": {
                "path": "source_locked.tar.gz",
                "sha256": sha256_file(package_dir / "source_locked.tar.gz"),
                "size_bytes": (
                    package_dir / "source_locked.tar.gz"
                ).stat().st_size,
                "manifest_sha256": source_manifest["sha256"],
            },
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
            },
            "logical_cell_count": 20,
            "direct_execution_count": 18,
            "files": [
                _manifest_file_binding(package_dir, relative)
                for relative in sorted(relative_files)
            ],
            "mutable_runtime_directories": list(MUTABLE_RUNTIME_DIRS),
            "scope": (
                "package_only; source_locked.tar.gz is separately hash-bound "
                "and contains only verified implementation inventory files "
                "plus the two immutable v7 bundle directories"
            ),
            "execution_authorized": True,
            "submission_authorized": True,
            "submission_state": "not_submitted",
        }
    )


def build_package(
    *,
    repo_root: Path,
    v7_root: Path | None,
    authorization_path: Path,
    validate_only: bool,
) -> dict[str, Any]:
    if (
        not authorization_path.is_file()
        or authorization_path.is_symlink()
    ):
        raise PackageContractError(
            "Authorization receipt must be a regular, non-symlinked file."
        )
    authority = validate_v7_authority(repo_root, v7_root=v7_root)
    control_plane = package_control_plane_receipt(PACKAGE_DIR)
    authorization = load_json_object(
        authorization_path, label="execution authorization receipt"
    )
    validate_authorization_receipt(
        authorization,
        v7_authority=authority,
        package_control_plane_sha256=control_plane["sha256"],
    )
    if validate_only:
        return {
            "status": "passed",
            "mode": "validate-only",
            "package_id": PACKAGE_ID,
            "v7_final_receipt_sha256": authority["final_receipt"]["sha256"],
            "study1_objective_gate_authority_sha256": authority[
                "objective_gate_authority"
            ]["sha256"],
            "authorization_sha256": authorization["sha256"],
            "package_control_plane_sha256": control_plane["sha256"],
            "logical_cell_count": 20,
            "direct_execution_count": 18,
            "writes_performed": False,
        }

    package_dir = PACKAGE_DIR
    authorization_destination = package_dir / EXECUTION_AUTHORITY_COPY
    authorization_is_prepositioned = (
        authorization_path.resolve() == authorization_destination.resolve()
    )
    generated_targets = [
        package_dir / "source_locked.tar.gz",
        package_dir / "package_manifest.json",
        *(package_dir / relative for relative in GENERATED_TOP_LEVEL_FILES),
        package_dir / V7_FINAL_AUTHORITY_COPY,
        package_dir / OBJECTIVE_GATE_AUTHORITY_COPY,
        *(
            ()
            if authorization_is_prepositioned
            else (authorization_destination,)
        ),
        *(package_dir / relative for relative in MUTABLE_RUNTIME_DIRS),
    ]
    jobs_dir = package_dir / "jobs"
    if any(path.exists() for path in generated_targets) or jobs_dir.exists():
        raise PackageContractError(
            "Generated package state already exists. Refusing an in-place "
            "rebuild; validate or preserve the existing immutable package."
        )
    for relative in STATIC_PACKAGE_FILES:
        path = package_dir / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Static package implementation is incomplete: {path}"
            )
    wrapper = package_dir / "execute_source_locked_job.sh"
    if not wrapper.stat().st_mode & 0o111:
        raise PackageContractError(
            "execute_source_locked_job.sh must be executable before sealing "
            "the package."
        )

    archive_members = _collect_archive_members(
        repo_root=repo_root, authority=authority
    )
    archive_path = package_dir / "source_locked.tar.gz"
    _build_deterministic_archive(
        repo_root=repo_root,
        destination=archive_path,
        members=archive_members,
    )
    source_manifest = digested(
        {
            "schema": "paper_i_ra_adapt_source_archive_manifest_v1",
            "package_id": PACKAGE_ID,
            "materialization_revision": V7_REVISION,
            "implementation_inventory_sha256": (
                authority["source_inventory"]["sha256"]
            ),
            "bundle_roots": [
                authority["bundle_bindings"][bundle_id]["bundle_root"]
                for bundle_id in BUNDLE_IDS
            ],
            "member_count": len(archive_members),
            "members": archive_members,
            "archive": {
                "path": "source_locked.tar.gz",
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
        }
    )

    final_source = Path(authority["v7_root"]) / "final_materialization_receipt.json"
    objective_source = (
        Path(authority["v7_root"])
        / "study1_objective_gate_authority_receipt.json"
    )
    final_copy = package_dir / V7_FINAL_AUTHORITY_COPY
    objective_copy = package_dir / OBJECTIVE_GATE_AUTHORITY_COPY
    authorization_copy = package_dir / EXECUTION_AUTHORITY_COPY
    try:
        _copy_exact(final_source, final_copy)
        _copy_exact(objective_source, objective_copy)
        if not authorization_is_prepositioned:
            _copy_exact(authorization_path, authorization_copy)
        atomic_write_json(
            package_dir / "source_archive_manifest.json", source_manifest
        )
        plan = _build_plan(
            repo_root=repo_root,
            authority=authority,
            authorization=authorization,
            authorization_file_sha256=sha256_file(authorization_path),
            source_manifest=source_manifest,
            source_archive_sha256=sha256_file(archive_path),
            control_plane=control_plane,
        )
        atomic_write_json(package_dir / "execution_plan.json", plan)

        jobs_dir.mkdir(parents=False, exist_ok=False)
        queue_lines: list[str] = []
        job_relatives: list[str] = []
        for row in plan["direct_executions"]:
            job = _build_job_spec(plan=plan, row=row)
            relative = f"jobs/{row['execution_id']}.json"
            atomic_write_json(package_dir / relative, job)
            job_relatives.append(relative)
            resources = row["resources"]
            queue_lines.append(
                "\t".join(
                    (
                        row["execution_id"],
                        relative,
                        sha256_file(package_dir / relative),
                        plan["source_archive"]["sha256"],
                        plan["authorization"]["file_sha256"],
                        str(resources["request_memory_mb"]),
                        str(resources["request_disk_mb"]),
                    )
                )
            )
        _exclusive_write_bytes(
            package_dir / "queue.tsv",
            ("\n".join(queue_lines) + "\n").encode("utf-8"),
        )
        for relative in MUTABLE_RUNTIME_DIRS:
            (package_dir / relative).mkdir(parents=False, exist_ok=False)
        package_manifest = _build_package_manifest(
            package_dir=package_dir,
            plan=plan,
            authority=authority,
            authorization=authorization,
            source_manifest=source_manifest,
            job_relatives=job_relatives,
            control_plane=control_plane,
        )
        atomic_write_json(
            package_dir / "package_manifest.json", package_manifest
        )
    except Exception:
        # Keep any already-written authority-bound artifacts for diagnosis.
        # The immutable no-overwrite rule forces an explicit human decision
        # before retrying a partial build.
        raise

    return {
        "status": "passed",
        "mode": "build",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": package_manifest["sha256"],
        "source_archive_sha256": source_manifest["archive"]["sha256"],
        "logical_cell_count": 20,
        "direct_execution_count": 18,
        "submission_state": "not_submitted",
    }


def _local_packaged_smoke(
    *,
    package_dir: Path,
    execution_id: str,
) -> None:
    """Explicit opt-in local execution using only the packaged source tree."""

    job_path = package_dir / "jobs" / f"{execution_id}.json"
    if not job_path.is_file():
        raise PackageContractError(f"Unknown packaged execution ID: {execution_id}")
    with tempfile.TemporaryDirectory(prefix="paper_i_study1_smoke_") as raw:
        root = Path(raw)
        with tarfile.open(package_dir / "source_locked.tar.gz", "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = safe_relative_path(
                    member.name, label="local smoke archive member"
                )
                if not member.isfile():
                    raise PackageContractError(
                        "Local smoke source archive contains a non-file member."
                    )
                target = root.joinpath(*relative.parts)
                if target.exists():
                    raise PackageContractError(
                        f"Local smoke archive collision: {member.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PackageContractError(
                        f"Cannot read local smoke member: {member.name}"
                    )
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
        job_relative = f"jobs/{execution_id}.json"
        staged_package_dir = stage_packaged_runtime_tree(
            package_dir=package_dir,
            source_root=root,
            job_relative=job_relative,
        )
        command = [
            sys.executable,
            str(staged_package_dir / "run_cell.py"),
            "--mode",
            "local-packaged-smoke",
            "--source-root",
            str(root),
            "--job-spec",
            str(staged_package_dir / job_relative),
            "--package-manifest",
            str(staged_package_dir / "package_manifest.json"),
            "--authorization-receipt",
            str(
                staged_package_dir
                / "authority/execution_authorization_receipt.json"
            ),
            "--v7-final-receipt",
            str(
                staged_package_dir
                / V7_FINAL_AUTHORITY_COPY
            ),
            "--objective-gate-authority",
            str(staged_package_dir / OBJECTIVE_GATE_AUTHORITY_COPY),
            "--execution-plan",
            str(staged_package_dir / "execution_plan.json"),
            "--source-archive-sha256",
            sha256_file(package_dir / "source_locked.tar.gz"),
        ]
        subprocess.run(command, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(__file__),
    )
    parser.add_argument("--v7-root", type=Path)
    parser.add_argument(
        "--authorization-receipt",
        type=Path,
        required=True,
        help="Externally minted, self-digested Study-1 authorization receipt.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate v7 and authorization without writing package artifacts.",
    )
    parser.add_argument(
        "--local-packaged-smoke",
        metavar="EXECUTION_ID",
        help=(
            "After a successful build, explicitly execute one authorized cell "
            "from an extracted packaged source tree. Never implied by build."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        receipt = build_package(
            repo_root=args.repo_root.resolve(),
            v7_root=(
                None if args.v7_root is None else args.v7_root.resolve()
            ),
            authorization_path=args.authorization_receipt.expanduser().absolute(),
            validate_only=bool(args.validate_only),
        )
        if args.local_packaged_smoke:
            if args.validate_only:
                raise PackageContractError(
                    "--local-packaged-smoke cannot be combined with "
                    "--validate-only."
                )
            _local_packaged_smoke(
                package_dir=PACKAGE_DIR,
                execution_id=args.local_packaged_smoke,
            )
            receipt["local_packaged_smoke"] = args.local_packaged_smoke
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (PackageContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
