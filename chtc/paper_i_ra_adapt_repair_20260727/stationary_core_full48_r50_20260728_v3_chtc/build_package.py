#!/usr/bin/env python3
"""Seal the inert, preauthorization 48-cell stationary-core CHTC package.

The builder requires a passed semantic P3 receipt but deliberately requires
the submission-authorization overlay to be absent.  It never runs a facade,
stages remote bytes, or calls HTCondor.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CONTROL_PLANE_FILES,
    CORE_FINAL_COPY_RELATIVE,
    EXECUTION_PLAN_SCHEMA,
    EXECUTION_TARGET,
    EXPECTED_ARTIFACT_ROLES,
    JOB_SPEC_SCHEMA,
    MUTABLE_RUNTIME_DIRECTORIES,
    P2_RECEIPT_RELATIVE,
    P2_RECEIPT_SCHEMA,
    P3_RECEIPT_RELATIVE,
    P4_RECEIPT_RELATIVE,
    P4_SMOKE_SPEC_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    PACKAGE_PREAUTHORIZATION_RELATIVE,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    RUNTIME_RELATIVE_ROOT,
    RUN_CLASS,
    SOURCE_ARCHIVE_MANIFEST_SCHEMA,
    SUBMISSION_AUTHORIZATION_RELATIVE,
    USER_SELECTION_COPY_RELATIVE,
    PackageContractError,
    atomic_publish_noreplace,
    atomic_write_json,
    canonical_json_bytes,
    control_plane_receipt,
    digested,
    direct_execution_ids,
    direct_execution_rows,
    expected_artifact_path,
    load_json_object,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_core_authority,
    validate_p3_receipt,
    validate_user_selection_authority,
)


GENERATED_FILES = (
    "control_plane_receipt.json",
    "source_archive_manifest.json",
    "source_locked.tar.gz",
    "execution_plan.json",
    "p4_smoke_spec.json",
    "queue.tsv",
    CORE_FINAL_COPY_RELATIVE,
    USER_SELECTION_COPY_RELATIVE,
    P2_RECEIPT_RELATIVE,
    P3_RECEIPT_RELATIVE,
)
GENERATED_DIRECTORIES = (
    "authority",
    "jobs",
    *MUTABLE_RUNTIME_DIRECTORIES,
)


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite stale temporary: {temporary}"
        )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_publish_noreplace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _copy_exact(source: Path, destination: Path) -> None:
    _exclusive_write(destination, source.read_bytes())


def _source_members(
    *,
    repo_root: Path,
    authority: Mapping[str, Any],
    user_selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for source_kind, rows in (
        ("verified_implementation_inventory", authority["source_files"]),
        (
            "verified_global_source_locks",
            authority["global_source_files"].values(),
        ),
        ("immutable_core_bundle", authority["bundle_files"]),
        (
            "core_final_publication_authority",
            [authority["final_receipt_binding"]],
        ),
        (
            "explicit_user_selection_authority",
            [user_selection["binding"]],
        ),
    ):
        for raw in rows:
            relative = safe_relative_path(
                raw["path"], label="source archive member"
            ).as_posix()
            binding = {
                "path": relative,
                "sha256": str(raw["sha256"]),
                "size_bytes": int(raw["size_bytes"]),
                "source_kind": source_kind,
            }
            previous = members.get(relative)
            if previous is not None:
                if (
                    previous["sha256"] != binding["sha256"]
                    or previous["size_bytes"] != binding["size_bytes"]
                ):
                    raise PackageContractError(
                        f"Source archive collision: {relative}"
                    )
                previous["source_kind"] = (
                    f"{previous['source_kind']}+{source_kind}"
                )
                continue
            source = repo_root / relative
            if (
                not source.is_file()
                or source.is_symlink()
                or sha256_file(source) != binding["sha256"]
                or source.stat().st_size != binding["size_bytes"]
            ):
                raise PackageContractError(
                    f"Source archive input drifted: {relative}"
                )
            members[relative] = binding
    return [members[key] for key in sorted(members)]


def _write_deterministic_archive(
    *,
    repo_root: Path,
    destination: Path,
    members: Iterable[Mapping[str, Any]],
) -> None:
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite source archive: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise PackageContractError(
            f"Refusing stale source archive temporary: {temporary}"
        )
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for row in members:
                        relative = safe_relative_path(
                            row["path"], label="archive member"
                        ).as_posix()
                        source = repo_root / relative
                        if (
                            not source.is_file()
                            or source.is_symlink()
                            or sha256_file(source) != row["sha256"]
                            or source.stat().st_size
                            != int(row["size_bytes"])
                        ):
                            raise PackageContractError(
                                f"Archive member drifted: {relative}"
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
        atomic_publish_noreplace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _p2_receipt(
    *,
    authority: Mapping[str, Any],
    user_selection: Mapping[str, Any],
    p3: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(direct_execution_rows())
    return digested(
        {
            "schema": P2_RECEIPT_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "status": "passed",
            "p2_passed": True,
            "core_final_receipt": dict(
                authority["final_receipt_binding"]
            ),
            "user_selection_authority": dict(
                user_selection["binding"]
            ),
            "implementation_source_inventory_sha256": authority[
                "implementation_inventory_sha256"
            ],
            "global_source_files": {
                key: dict(value)
                for key, value in sorted(
                    authority["global_source_files"].items()
                )
            },
            "bundle_manifest": dict(
                authority["document_bindings"]["bundle_manifest.json"]
            ),
            "source_locks": dict(
                authority["document_bindings"]["source_locks.json"]
            ),
            "expected_artifacts": dict(
                authority["document_bindings"]["expected_artifacts.json"]
            ),
            "validation_report": dict(
                authority["document_bindings"]["validation_report.json"]
            ),
            "six_regime_pool_construction_proof": dict(
                p3["p2_pool_construction_proof"]
            ),
            "six_regime_pool_construction_proof_sha256": p3[
                "p2_pool_construction_proof_sha256"
            ],
            "p3_receipt_sha256": p3["sha256"],
            "direct_cell_count": 48,
            "protocol_count": len(authority["protocol_bindings"]),
            "execution_template_count": len(
                authority["template_bindings"]
            ),
            "execution_ids": list(direct_execution_ids()),
            "regime_cutoff_pairs": [
                [row["regime_id"], row["nph"]]
                for row in rows[::8]
            ],
            "route_families": sorted(
                {str(row["route_id"]) for row in rows}
            ),
            "candidate_representations": sorted(
                {
                    str(row["candidate_representation"])
                    for row in rows
                }
            ),
            "full_horizon": 50,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "same_cutoff_reference_required": True,
            "active_gradient_policy": (
                "stationary_source_response_v1"
            ),
            "resource_weighting_scope": "late_resource_weighting_v1",
            "recursive_core_allowlist_passed": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
        }
    )


def _job_spec(
    *,
    plan_sha256: str,
    row: Mapping[str, Any],
    authority: Mapping[str, Any],
    archive_sha256: str,
    control_plane_sha256: str,
) -> dict[str, Any]:
    cell_id = str(row["cell_id"])
    bundle_root = (
        f"{authority['bundle_root']}"
    )
    repo_root = repo_root_from_script(__file__).as_posix()
    try:
        bundle_relative = Path(bundle_root).relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise PackageContractError(
            "Core bundle root escapes the active repository."
        ) from exc
    return digested(
        {
            "schema": JOB_SPEC_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            **dict(row),
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "execution_plan_sha256": plan_sha256,
            "core_final_receipt_canonical_sha256": authority[
                "final_receipt_binding"
            ]["canonical_sha256"],
            "core_bundle_root": bundle_relative,
            "protocol": dict(authority["protocol_bindings"][cell_id]),
            "execution_template": dict(
                authority["template_bindings"][cell_id]
            ),
            "source_archive_sha256": archive_sha256,
            "package_control_plane_sha256": control_plane_sha256,
            "artifact_paths": {
                role: (
                    f"{bundle_relative}/"
                    f"{expected_artifact_path(cell_id, role)}"
                )
                for role in EXPECTED_ARTIFACT_ROLES
            },
            "worker_receipt_path": (
                f"worker_receipts/{cell_id}.json"
            ),
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_authorization_overlay": (
                SUBMISSION_AUTHORIZATION_RELATIVE
            ),
            "submission_state": "awaiting_explicit_user_authorization",
        }
    )


def _package_file_binding(package_dir: Path, relative: str) -> dict[str, Any]:
    path = package_dir / safe_relative_path(
        relative, label="package file"
    )
    if not path.is_file() or path.is_symlink():
        raise PackageContractError(f"Package file is unavailable: {path}")
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "executable": bool(path.stat().st_mode & 0o111),
    }


def build_package(
    *,
    repo_root: Path,
    core_root: Path | None,
    p3_receipt_path: Path,
) -> dict[str, Any]:
    package_dir = PACKAGE_DIR
    for relative in CONTROL_PLANE_FILES:
        path = package_dir / relative
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Control-plane file is unavailable: {path}"
            )
    if not (package_dir / "execute_source_locked_job.sh").stat().st_mode & 0o111:
        raise PackageContractError(
            "execute_source_locked_job.sh must be executable before sealing."
        )
    generated_targets = [
        package_dir / relative
        for relative in (
            *GENERATED_FILES,
            "package_manifest.json",
            P4_RECEIPT_RELATIVE,
            PACKAGE_PREAUTHORIZATION_RELATIVE,
            SUBMISSION_AUTHORIZATION_RELATIVE,
        )
    ]
    generated_targets.extend(
        package_dir / relative for relative in GENERATED_DIRECTORIES
    )
    collisions = [str(path) for path in generated_targets if path.exists()]
    if collisions:
        raise PackageContractError(
            "Refusing an in-place package rebuild; existing targets: "
            + ", ".join(collisions)
        )
    if not p3_receipt_path.is_file() or p3_receipt_path.is_symlink():
        raise PackageContractError("P3 receipt is unavailable or unsafe.")

    authority = validate_core_authority(
        repo_root,
        materialization_root=core_root,
    )
    user_selection = validate_user_selection_authority(repo_root)
    control_plane = control_plane_receipt(package_dir)
    p3 = load_json_object(p3_receipt_path, label="P3 preflight receipt")
    p3_binding = validate_p3_receipt(
        p3,
        receipt_file_sha256=sha256_file(p3_receipt_path),
        authority=authority,
        control_plane=control_plane,
    )
    p2 = _p2_receipt(
        authority=authority,
        user_selection=user_selection,
        p3=p3,
    )

    for relative in GENERATED_DIRECTORIES:
        (package_dir / relative).mkdir(parents=True, exist_ok=False)
    _copy_exact(
        Path(authority["core_root"]) / "final_publication_receipt.json",
        package_dir / CORE_FINAL_COPY_RELATIVE,
    )
    _copy_exact(
        repo_root / user_selection["binding"]["path"],
        package_dir / USER_SELECTION_COPY_RELATIVE,
    )
    atomic_write_json(package_dir / P2_RECEIPT_RELATIVE, p2)
    _copy_exact(p3_receipt_path, package_dir / P3_RECEIPT_RELATIVE)
    atomic_write_json(
        package_dir / "control_plane_receipt.json", control_plane
    )

    members = _source_members(
        repo_root=repo_root,
        authority=authority,
        user_selection=user_selection,
    )
    archive_path = package_dir / "source_locked.tar.gz"
    _write_deterministic_archive(
        repo_root=repo_root,
        destination=archive_path,
        members=members,
    )
    source_manifest = digested(
        {
            "schema": SOURCE_ARCHIVE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "member_count": len(members),
            "members": members,
            "implementation_source_inventory_sha256": authority[
                "implementation_inventory_sha256"
            ],
            "core_final_receipt": dict(
                authority["final_receipt_binding"]
            ),
            "user_selection_authority": dict(
                user_selection["binding"]
            ),
            "archive": {
                "path": "source_locked.tar.gz",
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
        }
    )
    atomic_write_json(
        package_dir / "source_archive_manifest.json",
        source_manifest,
    )

    plan_unsigned = {
        "schema": EXECUTION_PLAN_SCHEMA,
        "package_id": PACKAGE_ID,
        "campaign_id": CAMPAIGN_ID,
        "run_class": RUN_CLASS,
        "execution_target": EXECUTION_TARGET,
        "core_final_receipt": dict(
            authority["final_receipt_binding"]
        ),
        "user_selection_authority": dict(
            user_selection["binding"]
        ),
        "p2_receipt": {
            "path": P2_RECEIPT_RELATIVE,
            "canonical_sha256": p2["sha256"],
            "file_sha256": sha256_file(
                package_dir / P2_RECEIPT_RELATIVE
            ),
        },
        "p3_receipt": dict(p3_binding),
        "package_control_plane": dict(control_plane),
        "source_archive": dict(source_manifest["archive"]),
        "source_archive_manifest_sha256": source_manifest["sha256"],
        "remote_image": {
            "path": REMOTE_IMAGE_PATH,
            "sha256": REMOTE_IMAGE_SHA256,
            "byte_verification_state": "pending_remote_pre_submit",
            "verification_must_pass_before_condor_submit": True,
        },
        "runtime_output_root": RUNTIME_RELATIVE_ROOT,
        "direct_execution_count": 48,
        "execution_ids": list(direct_execution_ids()),
        "g11_bounded_replay_diagnostic_count": 12,
        "g11_bounded_replay_diagnostic_execution_ids": [
            row["execution_id"]
            for row in direct_execution_rows()
            if row["g11_bounded_replay_diagnostic"]["selected"]
        ],
        "direct_executions": [],
        "shared_execution_count": 0,
        "append_dedupe_active": False,
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_authorization_overlay": {
            "path": SUBMISSION_AUTHORIZATION_RELATIVE,
            "required_before_condor_submit": True,
            "present": False,
        },
        "submission_state": "awaiting_explicit_user_authorization",
    }
    queue_lines: list[str] = []
    # The plan owns job hashes, so finalize it once and then bind that stable
    # digest into byte-identical regenerated specs.
    plan = digested(
        {
            **plan_unsigned,
            "direct_executions": [
                {
                    **dict(row),
                    "job_spec_path": f"jobs/{row['execution_id']}.json",
                }
                for row in direct_execution_rows()
            ],
        }
    )
    jobs = [
        _job_spec(
            plan_sha256=plan["sha256"],
            row=row,
            authority=authority,
            archive_sha256=source_manifest["archive"]["sha256"],
            control_plane_sha256=control_plane["sha256"],
        )
        for row in direct_execution_rows()
    ]
    atomic_write_json(package_dir / "execution_plan.json", plan)
    for row, job in zip(direct_execution_rows(), jobs, strict=True):
        relative = f"jobs/{row['execution_id']}.json"
        atomic_write_json(package_dir / relative, job)
        resources = row["resources"]
        queue_lines.append(
            "\t".join(
                (
                    str(row["execution_id"]),
                    relative,
                    sha256_file(package_dir / relative),
                    source_manifest["archive"]["sha256"],
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                )
            )
        )
    _exclusive_write(
        package_dir / "queue.tsv",
        ("\n".join(queue_lines) + "\n").encode("utf-8"),
    )
    p4_cell = "core__strong_weak_u8__nph3__ra_macro_append_only"
    p4_smoke = digested(
        {
            "schema": P4_SMOKE_SPEC_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "source_execution_id": p4_cell,
            "source_job_spec_path": f"jobs/{p4_cell}.json",
            "maximum_controller_rounds": 1,
            "run_class": "smoke",
            "purpose": (
                "bounded_packaged_dispatch_and_verification_only_v1"
            ),
            "paper_facing_result_allowed": False,
            "submission_authorized": False,
        }
    )
    atomic_write_json(package_dir / "p4_smoke_spec.json", p4_smoke)

    package_files = [
        *CONTROL_PLANE_FILES,
        *GENERATED_FILES,
        *(f"jobs/{execution_id}.json" for execution_id in direct_execution_ids()),
    ]
    # source_locked.tar.gz already appears in GENERATED_FILES; enforce exact
    # membership instead of silently deduplicating a malformed list.
    if len(package_files) != len(set(package_files)):
        raise PackageContractError("Package file inventory contains duplicates.")
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_plan_sha256": plan["sha256"],
            "package_control_plane_sha256": control_plane["sha256"],
            "source_archive": dict(source_manifest["archive"]),
            "source_archive_manifest_sha256": source_manifest["sha256"],
            "core_final_receipt": dict(
                authority["final_receipt_binding"]
            ),
            "user_selection_authority": dict(
                user_selection["binding"]
            ),
            "p2_receipt_sha256": p2["sha256"],
            "p3_receipt_sha256": p3["sha256"],
            "p4_smoke_spec_sha256": p4_smoke["sha256"],
            "remote_image": {
                "path": REMOTE_IMAGE_PATH,
                "sha256": REMOTE_IMAGE_SHA256,
                "byte_verification_state": "pending_remote_pre_submit",
                "verification_must_pass_before_condor_submit": True,
            },
            "runtime_output_root": RUNTIME_RELATIVE_ROOT,
            "direct_execution_count": 48,
            "shared_execution_count": 0,
            "append_dedupe_active": False,
            "files": [
                _package_file_binding(package_dir, relative)
                for relative in sorted(package_files)
            ],
            "mutable_runtime_directories": list(
                MUTABLE_RUNTIME_DIRECTORIES
            ),
            "declared_post_seal_overlays": {
                P4_RECEIPT_RELATIVE: {
                    "required_for_final_preauthorization_state": True,
                    "must_bind_package_manifest": True,
                },
                PACKAGE_PREAUTHORIZATION_RELATIVE: {
                    "required_for_final_preauthorization_state": True,
                    "must_bind_p4_receipt": True,
                },
                SUBMISSION_AUTHORIZATION_RELATIVE: {
                    "required_before_condor_submit": True,
                    "must_be_absent_before_explicit_user_authorization": True,
                },
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "remote_stage": False,
            "condor_submit": False,
            "submission_state": "awaiting_explicit_user_authorization",
        }
    )
    atomic_write_json(package_dir / "package_manifest.json", manifest)
    return {
        "status": "passed",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "source_archive_sha256": source_manifest["archive"]["sha256"],
        "p2_receipt_sha256": p2["sha256"],
        "p3_receipt_sha256": p3["sha256"],
        "direct_execution_count": 48,
        "p4_pending": True,
        "submission_authorization_overlay_present": False,
        "remote_stage": False,
        "condor_submit": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(__file__),
    )
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--p3-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = build_package(
            repo_root=args.repo_root.resolve(),
            core_root=(
                None if args.core_root is None else args.core_root.resolve()
            ),
            p3_receipt_path=args.p3_receipt.resolve(),
        )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except (PackageContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
