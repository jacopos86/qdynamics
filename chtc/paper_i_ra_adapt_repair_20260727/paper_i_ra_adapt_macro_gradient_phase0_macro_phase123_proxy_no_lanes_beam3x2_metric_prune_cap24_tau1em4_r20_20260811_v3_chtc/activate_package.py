#!/usr/bin/env python3
"""Prepare, but never submit, an explicitly authorized package activation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
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
    BUNDLE_ID,
    CAMPAIGN_ID,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    PackageContractError,
    binding,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    repo_root_from_script,
    sha256_file,
    verify_self_digest,
)
from run_cell import _load_closed_job, _validate_authorization  # noqa: E402
from validate_package import validate_package  # noqa: E402


REPO_ROOT = repo_root_from_script(__file__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise PackageContractError("Activation request is missing or unsafe.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256_file(source) != sha256_file(destination):
        raise PackageContractError("Activation-request copy drifted.")


def _relative_to_submit_root(path: Path, *, submit_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(submit_root.resolve())
    except ValueError as exc:
        raise PackageContractError(
            f"Activation path escaped the submit root: {path}."
        ) from exc
    if not relative.parts:
        raise PackageContractError("Submit-relative path cannot be empty.")
    text = relative.as_posix()
    if re.fullmatch(r"[A-Za-z0-9_./-]+", text) is None:
        raise PackageContractError(
            f"Submit-relative path contains unsupported characters: {text!r}."
        )
    return text


def _validate_request(
    path: Path,
    *,
    package_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    request = load_json(path, label="explicit activation request")
    verify_self_digest(request, label="explicit activation request")
    if (
        request.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or request.get("package_id") != PACKAGE_ID
        or request.get("campaign_id") != CAMPAIGN_ID
        or request.get("bundle_id") != BUNDLE_ID
        or request.get("package_manifest_sha256")
        != package_manifest.get("sha256")
        or request.get("requested_execution_ids")
        != list(expected_execution_ids())
        or request.get("scope")
        != "prepare_six_cell_chtc_execution_and_submission_v1"
        or request.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or request.get("explicit_user_authority_recorded") is not True
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("paper_evidence_adoption_authorized") is not False
        or request.get("submitted") is not False
    ):
        raise PackageContractError(
            "Explicit activation request is stale, incomplete, or overbroad."
        )
    return request


def _render_submit_template(
    *,
    package_relative: str,
    activation_relative: str,
    source_archive_sha256: str,
) -> str:
    template_path = PACKAGE_DIR / "submit.sub.in"
    if not template_path.is_file() or template_path.is_symlink():
        raise PackageContractError("Inert submit template is missing or unsafe.")
    rendered = template_path.read_text(encoding="utf-8")
    replacements = {
        "__PACKAGE_REL__": package_relative,
        "__ACTIVATION_REL__": activation_relative,
        "__SOURCE_ARCHIVE_SHA256__": source_archive_sha256,
    }
    for token, value in replacements.items():
        if token not in rendered:
            raise PackageContractError(f"Submit template token is absent: {token}.")
        rendered = rendered.replace(token, value)
    if any(token in rendered for token in replacements):
        raise PackageContractError("Unresolved submit-template token remains.")
    for forbidden_queue_macro in ("request_memory_mb", "request_disk_mb"):
        if re.search(rf"\b{forbidden_queue_macro}\b", rendered):
            raise PackageContractError(
                "Submit queue variable would create a phantom slot-resource "
                f"requirement: {forbidden_queue_macro}."
            )
    return rendered


def activate(
    *,
    request_path: Path,
    image_path: Path,
    activation_dir: Path,
    submit_root: Path,
    container_runtime: str | None = None,
) -> dict[str, Any]:
    if not submit_root.is_dir() or submit_root.is_symlink():
        raise PackageContractError("Submit root is missing or unsafe.")
    activation_dir = activation_dir.resolve()
    if activation_dir.exists() or activation_dir.is_symlink():
        raise FileExistsError("Refusing to overwrite an activation directory.")
    try:
        activation_dir.relative_to(PACKAGE_DIR.resolve())
    except ValueError:
        pass
    else:
        raise PackageContractError("Activation directory cannot enter the package.")
    package_manifest = load_json(
        PACKAGE_DIR / "package_manifest.json", label="package manifest"
    )
    verify_self_digest(package_manifest, label="package manifest")
    if (
        package_manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or package_manifest.get("status") != "passed_inert_six_cells"
        or package_manifest.get("execution_authorized") is not False
        or package_manifest.get("submission_authorized") is not False
        or package_manifest.get("submission_ready") is not False
        or package_manifest.get("submitted") is not False
    ):
        raise PackageContractError("Package lost its inert activation boundary.")
    request = _validate_request(
        request_path.resolve(), package_manifest=package_manifest
    )
    if (
        not image_path.is_file()
        or image_path.is_symlink()
        or sha256_file(image_path) != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Pinned image is absent or hash-mismatched.")
    probe_receipt = validate_package(
        deep=True,
        image_path=image_path.resolve(),
        container_runtime=container_runtime,
        require_launch_ready=True,
    )
    if (
        probe_receipt.get("status") != "passed_inert_package"
        or probe_receipt.get("launch_ready") is not True
        or probe_receipt.get("deep_worker_preflight_count")
        != len(expected_execution_ids())
        or probe_receipt.get("deep_worker_preflight_runtime")
        != "pinned_execution_image"
        or probe_receipt.get("execution_authorized") is not False
        or probe_receipt.get("submission_authorized") is not False
    ):
        raise PackageContractError("Pinned-image launch probe did not pass.")

    activation_parent = activation_dir.parent
    if not activation_parent.is_dir() or activation_parent.is_symlink():
        raise PackageContractError("Activation parent is missing or unsafe.")
    package_relative = _relative_to_submit_root(
        PACKAGE_DIR, submit_root=submit_root
    )
    activation_relative = _relative_to_submit_root(
        activation_dir, submit_root=submit_root
    )
    source_archive = package_manifest.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise PackageContractError("Package source archive binding is absent.")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{activation_dir.name}.build-", dir=activation_parent)
    )
    try:
        request_copy = temporary / "activation_request.json"
        _copy_exact(request_path.resolve(), request_copy)
        probe_path = temporary / "image_runtime_probe.json"
        _write_json(probe_path, probe_receipt)
        request_binding = binding(request_copy, root=temporary, canonical=True)
        probe_binding = binding(probe_path, root=temporary, canonical=True)

        authorization_bindings: list[dict[str, Any]] = []
        for execution_id in expected_execution_ids():
            job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
            job, _manifest, _protocol, _locks = _load_closed_job(job_path)
            authority = digested(
                {
                    "schema": AUTHORIZATION_SCHEMA,
                    "package_id": PACKAGE_ID,
                    "campaign_id": CAMPAIGN_ID,
                    "bundle_id": BUNDLE_ID,
                    "execution_id": execution_id,
                    "job_spec_sha256": job["sha256"],
                    "package_manifest_sha256": package_manifest["sha256"],
                    "protocol_sha256": job["protocol_sha256"],
                    "source_archive_sha256": source_archive["sha256"],
                    "activation_request": request_binding,
                    "image_runtime_probe": probe_binding,
                    "pinned_image_path": REMOTE_IMAGE_PATH,
                    "pinned_image_sha256": REMOTE_IMAGE_SHA256,
                    "scope": "single_cell_chtc_execution_only",
                    "authorization_kind": (
                        "explicit_user_execution_and_submission_authority"
                    ),
                    "execution_authorized": True,
                    "submission_authorized": True,
                    "paper_evidence_adoption_authorized": False,
                    "submitted": False,
                }
            )
            authority_path = temporary / "authorizations" / f"{execution_id}.json"
            _write_json(authority_path, authority)
            _validate_authorization(
                authority_path,
                job=job,
                manifest=package_manifest,
            )
            authorization_bindings.append(
                {
                    "execution_id": execution_id,
                    **binding(
                        authority_path,
                        root=temporary,
                        canonical=True,
                    ),
                }
            )

        rendered = _render_submit_template(
            package_relative=package_relative,
            activation_relative=activation_relative,
            source_archive_sha256=str(source_archive["sha256"]),
        )
        submit_path = temporary / "submit.sub"
        with submit_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = digested(
            {
                "schema": ACTIVATION_MANIFEST_SCHEMA,
                "status": "passed_activation_prepared_no_submission",
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "package_manifest_sha256": package_manifest["sha256"],
                "activation_request": request_binding,
                "image_runtime_probe": probe_binding,
                "pinned_image_path": REMOTE_IMAGE_PATH,
                "pinned_image_sha256": REMOTE_IMAGE_SHA256,
                "authorizations": authorization_bindings,
                "authorization_count": len(authorization_bindings),
                "submit_descriptor": binding(submit_path, root=temporary),
                "package_relative_to_submit_root": package_relative,
                "activation_relative_to_submit_root": activation_relative,
                "launch_ready": True,
                "execution_authorized": True,
                "submission_authorized": True,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        )
        _write_json(temporary / "activation_manifest.json", manifest)
        os.rename(temporary, activation_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--submit-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--container-runtime")
    args = parser.parse_args()
    try:
        receipt = activate(
            request_path=args.request,
            image_path=args.image,
            activation_dir=args.activation_dir,
            submit_root=args.submit_root.resolve(),
            container_runtime=args.container_runtime,
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
