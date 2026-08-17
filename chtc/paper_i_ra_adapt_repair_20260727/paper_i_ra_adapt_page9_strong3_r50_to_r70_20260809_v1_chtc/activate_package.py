#!/usr/bin/env python3
"""Create a non-colliding, three-row activation after every resume input exists."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from package_contract import (  # noqa: E402
    ACTIVATION_MANIFEST_SCHEMA,
    ACTIVATION_REQUEST_SCHEMA,
    AUTHORIZATION_SCHEMA,
    BASE_PACKAGE_RELATIVE,
    CAMPAIGN_ID,
    PACKAGE_ID,
    REGIMES,
    REMOTE_IMAGE_RELATIVE,
    REMOTE_IMAGE_SHA256,
    RESOURCE_ENVELOPE,
    STAGING_OUTPUT_ROOT,
    PackageContractError,
    canonical_json_bytes,
    continuation_execution_id,
    digested,
    expected_execution_ids,
    file_binding,
    load_json,
    repo_root_from_script,
    safe_absolute_posix_path,
    sha256_file,
    verify_self_digest,
)
from run_cell import _load_job, _validate_materialization  # noqa: E402


REPO_ROOT = repo_root_from_script(__file__)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite activation file: {path}")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(path, canonical_json_bytes(value) + b"\n")


def _verified_request(path: Path, package_manifest: Mapping[str, Any]) -> dict[str, Any]:
    request = load_json(path, label="activation request")
    verify_self_digest(request, label="activation request")
    if (
        request.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or request.get("package_id") != PACKAGE_ID
        or request.get("campaign_id") != CAMPAIGN_ID
        or request.get("package_manifest_sha256") != package_manifest.get("sha256")
        or request.get("execution_ids") != list(expected_execution_ids())
        or request.get("scope") != "page9_strong_sector_r50_to_r70_three_cells"
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("paper_evidence_adoption_authorized") is not False
    ):
        raise PackageContractError("Activation request authority drifted.")
    return request


def _resume_paths(root: Path, execution_id: str) -> tuple[Path, Path]:
    cell_root = root / execution_id
    return (
        cell_root / "resume_materialization.json",
        cell_root / "resume_input.tar.gz",
    )


def _pinned_image_runtime_preflight() -> dict[str, Any]:
    image = REPO_ROOT / REMOTE_IMAGE_RELATIVE
    if (
        not image.is_file()
        or image.is_symlink()
        or sha256_file(image) != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Pinned remote image bytes are unavailable or drifted.")
    runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise PackageContractError("Apptainer/Singularity is unavailable for image preflight.")
    code = (
        "import io,sys;"
        f"sys.path.insert(0,{PACKAGE_DIR.as_posix()!r});"
        "from vendored_ijson_python import BACKEND,VENDORED_IJSON_VERSION,parse;"
        "events=list(parse(io.BytesIO(b'{\"x\":[1,true,null]}'),buf_size=2));"
        "assert BACKEND=='python' and VENDORED_IJSON_VERSION=='3.5.1';"
        "assert ('x.item','number',1) in events"
    )
    completed = subprocess.run(
        [
            runtime,
            "exec",
            "--cleanenv",
            "--bind",
            f"{PACKAGE_DIR}:{PACKAGE_DIR}:ro",
            image.as_posix(),
            "python",
            "-I",
            "-S",
            "-B",
            "-c",
            code,
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            "Vendored streaming parser failed in the pinned image: "
            + completed.stderr.strip()
        )
    return digested(
        {
            "schema": "paper_i_page9_r70_pinned_image_runtime_preflight_v1",
            "status": "passed",
            "image": {
                "path": REMOTE_IMAGE_RELATIVE.as_posix(),
                "sha256": REMOTE_IMAGE_SHA256,
                "size_bytes": image.stat().st_size,
            },
            "runtime": Path(runtime).name,
            "python_flags": ["-I", "-S", "-B"],
            "vendored_distribution": "ijson",
            "vendored_version": "3.5.1",
            "backend": "python",
            "ambient_site_packages_disabled": True,
        }
    )


def activate(
    *, request_path: Path, resume_root: Path, output_dir: Path
) -> dict[str, Any]:
    package_manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(package_manifest, label="package manifest")
    request = _verified_request(request_path, package_manifest)
    resume_root = safe_absolute_posix_path(
        resume_root.as_posix(), label="resume materialization root"
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise PackageContractError(f"Refusing existing activation root: {output_dir}")
    try:
        output_relative = output_dir.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise PackageContractError("Activation root must be inside the active repository.") from exc

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for regime in REGIMES:
        execution_id = continuation_execution_id(regime)
        manifest_path, archive_path = _resume_paths(resume_root, execution_id)
        if not manifest_path.is_file() or not archive_path.is_file():
            missing.append(execution_id)
            continue
        job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
        job, _manifest, _protocol = _load_job(job_path)
        materialization = _validate_materialization(
            job=job,
            manifest_path=manifest_path,
            archive_path=archive_path,
        )
        rows.append(
            {
                "regime_id": regime,
                "execution_id": execution_id,
                "job": job,
                "job_path": job_path,
                "resume_manifest_path": manifest_path,
                "resume_archive_path": archive_path,
                "resume_materialization": materialization,
            }
        )
    if missing:
        raise PackageContractError(
            "Activation is blocked by missing authenticated resume inputs: "
            + ", ".join(missing)
        )
    if [row["execution_id"] for row in rows] != list(expected_execution_ids()):
        raise PackageContractError("Activation row order/closure drifted.")
    image_preflight = _pinned_image_runtime_preflight()

    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        request_copy = output_dir / "activation_request.json"
        shutil.copyfile(request_path, request_copy)
        authorizations: list[dict[str, Any]] = []
        queue_lines: list[str] = []
        for row in rows:
            job = row["job"]
            materialization = row["resume_materialization"]
            execution_id = row["execution_id"]
            authorization = digested(
                {
                    "schema": AUTHORIZATION_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "execution_id": execution_id,
                    "scope": "single_page9_strong_sector_r70_cell",
                    "activation_request_sha256": request["sha256"],
                    "package_manifest_sha256": package_manifest["sha256"],
                    "job_sha256": job["sha256"],
                    "derived_protocol_sha256": job["derived_protocol_sha256"],
                    "resume_materialization_sha256": materialization["sha256"],
                    "resume_archive_sha256": materialization["archive"]["sha256"],
                    "pinned_image_sha256": REMOTE_IMAGE_SHA256,
                    "resources": dict(RESOURCE_ENVELOPE),
                    "execution_authorized": True,
                    "submission_authorized": True,
                    "paper_evidence_adoption_authorized": False,
                }
            )
            auth_relative = Path("authorizations") / f"{execution_id}.json"
            auth_path = output_dir / auth_relative
            _write_json(auth_path, authorization)
            authorizations.append(
                {"execution_id": execution_id, **file_binding(auth_path, relative_to=output_dir)}
            )
            archive_path = row["resume_archive_path"]
            manifest_path = row["resume_manifest_path"]
            queue_lines.append(
                "\t".join(
                    (
                        execution_id,
                        f"jobs/{execution_id}.json",
                        auth_relative.as_posix(),
                        archive_path.as_posix(),
                        manifest_path.as_posix(),
                        archive_path.name,
                        manifest_path.name,
                        materialization["archive"]["sha256"],
                        str(RESOURCE_ENVELOPE["request_cpus"]),
                        str(RESOURCE_ENVELOPE["request_memory_mb"]),
                        str(RESOURCE_ENVELOPE["request_disk_mb"]),
                        str(RESOURCE_ENVELOPE["max_runtime_seconds"]),
                    )
                )
            )
        queue_path = output_dir / "ready_queue.tsv"
        _write_bytes(queue_path, ("\n".join(queue_lines) + "\n").encode("utf-8"))

        package_relative = PACKAGE_DIR.relative_to(REPO_ROOT).as_posix()
        base_relative = BASE_PACKAGE_RELATIVE.as_posix()
        activation_relative = output_relative.as_posix()
        substitutions = {
            "@@PACKAGE_RELATIVE@@": package_relative,
            "@@PACKAGE_BASENAME@@": PACKAGE_DIR.name,
            "@@BASE_PACKAGE_RELATIVE@@": base_relative,
            "@@BASE_PACKAGE_BASENAME@@": BASE_PACKAGE_RELATIVE.name,
            "@@ACTIVATION_RELATIVE@@": activation_relative,
            "@@ACTIVATION_BASENAME@@": output_dir.name,
            "@@IMAGE_RELATIVE@@": REMOTE_IMAGE_RELATIVE.as_posix(),
            "@@IMAGE_BASENAME@@": REMOTE_IMAGE_RELATIVE.name,
            "@@IMAGE_SHA256@@": REMOTE_IMAGE_SHA256,
            "@@REMOTE_OUTPUT_ROOT@@": STAGING_OUTPUT_ROOT.as_posix(),
        }
        descriptor = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
        for token, value in substitutions.items():
            descriptor = descriptor.replace(token, value)
        if "@@" in descriptor or "output_destination" in descriptor:
            raise PackageContractError("Rendered submit descriptor is unsafe/incomplete.")
        submit_path = output_dir / "submit.sub"
        _write_bytes(submit_path, descriptor.encode("utf-8"))
        activation_manifest = digested(
            {
                "schema": ACTIVATION_MANIFEST_SCHEMA,
                "status": "passed_ready_not_submitted",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "activation_request": {
                    **file_binding(request_copy, relative_to=output_dir),
                    "canonical_sha256": request["sha256"],
                },
                "package_manifest_sha256": package_manifest["sha256"],
                "row_count": 3,
                "execution_ids": list(expected_execution_ids()),
                "authorizations": authorizations,
                "queue": file_binding(queue_path, relative_to=output_dir),
                "submit_descriptor": file_binding(submit_path, relative_to=output_dir),
                "resume_materialization_sha256s": {
                    row["execution_id"]: row["resume_materialization"]["sha256"]
                    for row in rows
                },
                "pinned_image_runtime_preflight": image_preflight,
                "explicit_transfer_output_files": True,
                "posix_staging_output_remaps": True,
                "failure_safe_attempt_capture": True,
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_ready": True,
                "submitted": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        )
        _write_json(output_dir / "activation_manifest.json", activation_manifest)
        return activation_manifest
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = activate(
            request_path=args.request.resolve(),
            resume_root=args.resume_root,
            output_dir=args.output_dir.resolve(),
        )
    except (OSError, ValueError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
