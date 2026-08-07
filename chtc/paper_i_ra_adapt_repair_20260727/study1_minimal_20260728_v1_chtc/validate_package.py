#!/usr/bin/env python3
"""Offline validator for the immutable Study-1 package.

The default mode is non-executing.  ``local-packaged-smoke`` is an explicit,
authorization-bound opt-in that executes exactly one direct job from a
temporary extraction of ``source_locked.tar.gz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    EXECUTION_PLAN_SCHEMA,
    JOB_SPEC_SCHEMA,
    MAX_RUNTIME_SECONDS,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_IMAGE_SHA256,
    REQUEST_CPUS,
    V7_RELATIVE_ROOT,
    PackageContractError,
    direct_execution_ids,
    load_json_object,
    logical_cell_keys,
    objective_gate_diagnostic_contract,
    package_control_plane_receipt,
    safe_relative_path,
    sha256_file,
    stage_packaged_runtime_tree,
    validate_authorization_receipt,
    validate_v7_authority,
    verify_exact_key_set,
    verify_self_digest,
)


def _safe_archive_rows(
    *,
    archive_path: Path,
    source_manifest: Mapping[str, Any],
) -> list[tarfile.TarInfo]:
    expected_rows = source_manifest.get("members")
    if (
        not isinstance(expected_rows, list)
        or int(source_manifest.get("member_count", -1)) != len(expected_rows)
    ):
        raise PackageContractError("Source archive manifest count drifted.")
    expected: dict[str, Mapping[str, Any]] = {}
    for row in expected_rows:
        if not isinstance(row, Mapping):
            raise PackageContractError("Source archive manifest row is invalid.")
        name = safe_relative_path(
            row.get("path"), label="source manifest member"
        ).as_posix()
        if name in expected:
            raise PackageContractError(
                f"Duplicate source manifest member: {name}"
            )
        expected[name] = row

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        observed: dict[str, tarfile.TarInfo] = {}
        for member in members:
            name = safe_relative_path(
                member.name, label="source archive member"
            ).as_posix()
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
                or name in observed
            ):
                raise PackageContractError(
                    f"Unsafe or duplicate source archive member: {member.name}"
                )
            observed[name] = member
        verify_exact_key_set(
            observed, tuple(expected), label="source archive members"
        )
        for name, member in observed.items():
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Cannot read source archive member: {name}"
                )
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            row = expected[name]
            if (
                digest.hexdigest() != row.get("sha256")
                or size != int(row.get("size_bytes", -1))
                or member.size != size
            ):
                raise PackageContractError(
                    f"Source archive member hash/size drifted: {name}"
                )
        return members


def _extract_safe(
    *,
    archive_path: Path,
    destination: Path,
) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            safe_relative_path(member.name, label="source extraction member")
            if (
                not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe source extraction member: {member.name}"
                )
        for member in members:
            relative = safe_relative_path(
                member.name, label="source extraction member"
            )
            target = destination.joinpath(*relative.parts)
            if target.exists():
                raise PackageContractError(
                    f"Source extraction collision: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise PackageContractError(
                    f"Cannot read source extraction member: {member.name}"
                )
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)


def validate_package(
    *,
    package_dir: Path,
    image_path: Path | None,
    extracted_root: Path,
) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    manifest = load_json_object(manifest_path, label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("package_id") != PACKAGE_ID
        or int(manifest.get("logical_cell_count", -1)) != 20
        or int(manifest.get("direct_execution_count", -1)) != 18
        or manifest.get("execution_authorized") is not True
        or manifest.get("submission_authorized") is not True
        or manifest.get("submission_state") != "not_submitted"
    ):
        raise PackageContractError("Package-manifest state drifted.")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise PackageContractError("Package manifest has no file inventory.")
    package_files: dict[str, Mapping[str, Any]] = {}
    for row in raw_files:
        if not isinstance(row, Mapping):
            raise PackageContractError("Package file row is invalid.")
        relative = safe_relative_path(
            row.get("path"), label="package-only path"
        ).as_posix()
        if relative in package_files:
            raise PackageContractError(
                f"Duplicate package-only path: {relative}"
            )
        path = package_dir / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or bool(path.stat().st_mode & 0o111)
            is not bool(row.get("executable"))
        ):
            raise PackageContractError(
                f"Package-only file drifted: {relative}"
            )
        package_files[relative] = row
    if manifest.get("mutable_runtime_directories") != ["fetched", "logs"]:
        raise PackageContractError("Mutable runtime directory contract drifted.")
    for relative in manifest["mutable_runtime_directories"]:
        path = package_dir / relative
        if not path.is_dir() or path.is_symlink():
            raise PackageContractError(
                f"Required runtime directory is unavailable: {relative}"
            )

    source_binding = manifest.get("source_archive")
    if not isinstance(source_binding, Mapping):
        raise PackageContractError("Package has no source archive binding.")
    archive_path = package_dir / safe_relative_path(
        source_binding.get("path"), label="source archive path"
    )
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or sha256_file(archive_path) != source_binding.get("sha256")
        or archive_path.stat().st_size
        != int(source_binding.get("size_bytes", -1))
    ):
        raise PackageContractError("Source archive binding drifted.")
    source_manifest = load_json_object(
        package_dir / "source_archive_manifest.json",
        label="source archive manifest",
    )
    verify_self_digest(source_manifest, label="source archive manifest")
    if (
        source_manifest.get("schema")
        != "paper_i_ra_adapt_source_archive_manifest_v1"
        or source_manifest.get("package_id") != PACKAGE_ID
        or source_manifest.get("sha256")
        != source_binding.get("manifest_sha256")
        or source_manifest.get("archive", {}).get("sha256")
        != source_binding.get("sha256")
    ):
        raise PackageContractError("Source archive manifest binding drifted.")
    _safe_archive_rows(
        archive_path=archive_path, source_manifest=source_manifest
    )
    _extract_safe(archive_path=archive_path, destination=extracted_root)

    final_path = (
        package_dir / "authority/v7_final_materialization_receipt.json"
    )
    objective_path = (
        package_dir
        / "authority/study1_objective_gate_authority_receipt.json"
    )
    authority = validate_v7_authority(
        extracted_root,
        v7_root=extracted_root / V7_RELATIVE_ROOT,
        final_receipt_path=final_path,
        objective_gate_authority_path=objective_path,
    )
    authorization_path = (
        package_dir / "authority/execution_authorization_receipt.json"
    )
    authorization = load_json_object(
        authorization_path, label="authorization receipt"
    )
    control_plane = package_control_plane_receipt(package_dir)
    validate_authorization_receipt(
        authorization,
        v7_authority=authority,
        package_control_plane_sha256=control_plane["sha256"],
    )
    if (
        manifest.get("authorization_sha256") != authorization["sha256"]
        or manifest.get("v7_final_receipt_sha256")
        != authority["final_receipt"]["sha256"]
        or manifest.get("study1_objective_gate_authority_sha256")
        != authority["objective_gate_authority"]["sha256"]
        or manifest.get("package_control_plane_sha256")
        != control_plane["sha256"]
    ):
        raise PackageContractError("Package authority hash drifted.")

    plan = load_json_object(
        package_dir / "execution_plan.json", label="execution plan"
    )
    verify_self_digest(plan, label="execution plan")
    logical = plan.get("logical_cells")
    direct = plan.get("direct_executions")
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("package_id") != PACKAGE_ID
        or plan.get("sha256") != manifest["execution_plan_sha256"]
        or not isinstance(logical, list)
        or not isinstance(direct, list)
        or [row.get("logical_key") for row in logical]
        != list(logical_cell_keys())
        or [row.get("execution_id") for row in direct]
        != list(direct_execution_ids())
        or len(plan.get("shared_append_references", ())) != 2
        or plan.get("study1_dedupe_sha256")
        != authority["dedupe_sha256"]
        or plan.get("package_control_plane") != control_plane
        or plan.get("source_inventory_sha256")
        != authority["source_inventory"]["sha256"]
        or plan.get("study1_objective_gate_authority", {}).get(
            "canonical_sha256"
        )
        != authority["objective_gate_authority"]["sha256"]
        or plan.get("study1_objective_gate_authority", {}).get(
            "file_sha256"
        )
        != authority["objective_gate_authority"]["file_sha256"]
        or plan.get("source_archive", {}).get("sha256")
        != source_binding["sha256"]
        or plan.get("remote_image", {}).get("sha256")
        != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Exact 20/18 execution plan drifted.")

    jobs: dict[str, Mapping[str, Any]] = {}
    for row in direct:
        execution_id = str(row["execution_id"])
        relative = f"jobs/{execution_id}.json"
        job = load_json_object(package_dir / relative, label="job spec")
        verify_self_digest(job, label=f"job spec {execution_id}")
        if (
            job.get("schema") != JOB_SPEC_SCHEMA
            or job.get("execution_id") != execution_id
            or job.get("execution_plan_sha256") != plan["sha256"]
            or job.get("protocol") != row.get("protocol")
            or job.get("execution_template")
            != row.get("execution_template")
            or job.get("artifact_paths") != row.get("artifact_paths")
            or job.get("resources") != row.get("resources")
            or job.get("objective_gate_diagnostics")
            != row.get("objective_gate_diagnostics")
            or job.get("objective_gate_diagnostics")
            != objective_gate_diagnostic_contract(
                bundle_id=str(row["bundle_id"]),
                regime_id=str(row["regime_id"]),
                route_id=str(row["route_id"]),
            )
            or job.get("package_control_plane") != control_plane
            or job.get("authorization") != plan["authorization"]
            or job.get("v7_final_receipt")
            != plan["v7_final_receipt"]
            or job.get("study1_objective_gate_authority")
            != plan["study1_objective_gate_authority"]
            or int(job.get("resources", {}).get("request_cpus", -1))
            != REQUEST_CPUS
            or int(
                job.get("resources", {}).get(
                    "max_runtime_seconds", -1
                )
            )
            != MAX_RUNTIME_SECONDS
        ):
            raise PackageContractError(
                f"Job spec/plan drifted: {execution_id}"
            )
        jobs[execution_id] = job

    queue_lines = (package_dir / "queue.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(queue_lines) != 18:
        raise PackageContractError("queue.tsv must contain exactly 18 rows.")
    queue_ids: list[str] = []
    for line in queue_lines:
        fields = line.split("\t")
        if len(fields) != 7:
            raise PackageContractError("queue.tsv row has the wrong arity.")
        (
            execution_id,
            job_relative,
            job_file_sha256,
            archive_sha256,
            authorization_file_sha256,
            memory_mb,
            disk_mb,
        ) = fields
        job = jobs.get(execution_id)
        if (
            job is None
            or job_relative != f"jobs/{execution_id}.json"
            or job_file_sha256
            != sha256_file(package_dir / job_relative)
            or archive_sha256 != source_binding["sha256"]
            or authorization_file_sha256
            != sha256_file(authorization_path)
            or int(memory_mb)
            != int(job["resources"]["request_memory_mb"])
            or int(disk_mb) != int(job["resources"]["request_disk_mb"])
        ):
            raise PackageContractError(
                f"queue.tsv row drifted: {execution_id}"
            )
        queue_ids.append(execution_id)
    if queue_ids != list(direct_execution_ids()):
        raise PackageContractError("queue.tsv execution order drifted.")

    image_status = "not_supplied"
    if image_path is not None:
        if (
            not image_path.is_file()
            or image_path.is_symlink()
            or sha256_file(image_path) != REMOTE_IMAGE_SHA256
        ):
            raise PackageContractError("Execution image hash drifted.")
        image_status = "verified"
    return {
        "status": "passed",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "execution_plan_sha256": plan["sha256"],
        "authorization_sha256": authorization["sha256"],
        "v7_final_receipt_sha256": authority["final_receipt"]["sha256"],
        "study1_objective_gate_authority_sha256": authority[
            "objective_gate_authority"
        ]["sha256"],
        "package_control_plane_sha256": control_plane["sha256"],
        "source_archive_sha256": source_binding["sha256"],
        "remote_image_sha256": REMOTE_IMAGE_SHA256,
        "remote_image_status": image_status,
        "logical_cell_count": 20,
        "direct_execution_count": 18,
        "shared_reference_count": 2,
        "writes_performed_outside_temporary_validation_root": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir", type=Path, default=PACKAGE_DIR
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--mode",
        choices=("validate-only", "local-packaged-smoke"),
        default="validate-only",
    )
    parser.add_argument(
        "--execution-id",
        help="Required only for explicit local-packaged-smoke mode.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.mode == "local-packaged-smoke":
            if args.execution_id not in direct_execution_ids():
                raise PackageContractError(
                    "--execution-id must name one of the exact 18 direct jobs."
                )
        elif args.execution_id is not None:
            raise PackageContractError(
                "--execution-id is only valid in local-packaged-smoke mode."
            )
        package_dir = args.package_dir.resolve()
        with tempfile.TemporaryDirectory(
            prefix="paper_i_study1_package_validation_"
        ) as raw:
            extracted = Path(raw)
            receipt = validate_package(
                package_dir=package_dir,
                image_path=(
                    None if args.image is None else args.image.resolve()
                ),
                extracted_root=extracted,
            )
            receipt["mode"] = args.mode
            if args.mode == "local-packaged-smoke":
                job_relative = f"jobs/{args.execution_id}.json"
                staged_package_dir = stage_packaged_runtime_tree(
                    package_dir=package_dir,
                    source_root=extracted,
                    job_relative=job_relative,
                )
                command = [
                    sys.executable,
                    str(staged_package_dir / "run_cell.py"),
                    "--mode",
                    "local-packaged-smoke",
                    "--source-root",
                    str(extracted),
                    "--job-spec",
                    str(
                        staged_package_dir / job_relative
                    ),
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
                        / "authority/v7_final_materialization_receipt.json"
                    ),
                    "--objective-gate-authority",
                    str(
                        staged_package_dir
                        / "authority/"
                        "study1_objective_gate_authority_receipt.json"
                    ),
                    "--execution-plan",
                    str(staged_package_dir / "execution_plan.json"),
                    "--source-archive-sha256",
                    receipt["source_archive_sha256"],
                ]
                subprocess.run(command, check=True)
                receipt["local_packaged_smoke_execution_id"] = (
                    args.execution_id
                )
                receipt[
                    "writes_performed_outside_temporary_validation_root"
                ] = False
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (PackageContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(
            f"ERROR: local packaged smoke failed with {exc.returncode}",
            file=sys.stderr,
        )
        return int(exc.returncode) or 2


if __name__ == "__main__":
    raise SystemExit(main())
