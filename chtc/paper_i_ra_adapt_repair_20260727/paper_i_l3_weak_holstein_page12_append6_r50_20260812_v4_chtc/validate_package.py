#!/usr/bin/env python3
"""Validate the inert package and optionally probe it in the pinned image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    BUNDLE_ID,
    BATCH_NAME,
    CAMPAIGN_ID,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REQUIRED_ROUTE_SOURCE_PATHS,
    RESOURCE_ENVELOPES,
    SCIENTIFIC_EQUIVALENCE_RELATIVE,
    STAGING_OUTPUT_ROOT,
    REMOTE_IMAGE_SHA256,
    V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256,
    V3_PACKAGE_MANIFEST_CANONICAL_SHA256,
    V3_SOURCE_ARCHIVE_SHA256,
    V3_SOURCE_ARCHIVE_SIZE_BYTES,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    sha256_file,
    verify_self_digest,
)
from run_cell import preflight  # noqa: E402


def _binding(raw: Any, *, label: str, canonical: bool = False) -> Path:
    if not isinstance(raw, Mapping):
        raise PackageContractError(f"{label} binding is absent.")
    path = PACKAGE_DIR / str(raw.get("path", ""))
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or sha256_file(path) != raw.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if canonical:
        payload = load_json(path, label=label)
        if verify_self_digest(payload, label=label) != raw.get(
            "canonical_sha256"
        ):
            raise PackageContractError(f"{label} canonical binding drifted.")
    return path


def _runtime(container_runtime: str | None) -> str:
    selected = container_runtime or shutil.which("apptainer") or shutil.which(
        "singularity"
    )
    if not selected:
        raise PackageContractError("Apptainer/Singularity is unavailable.")
    return str(selected)


def _inside_image(
    *,
    runtime: str,
    image_path: Path,
    arguments: list[str],
) -> dict[str, Any]:
    command = [
        runtime,
        "exec",
        "--cleanenv",
        "--bind",
        f"{PACKAGE_DIR.parent.parent.parent}:{PACKAGE_DIR.parent.parent.parent}",
        image_path.as_posix(),
        "python3",
        *arguments,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=PACKAGE_DIR,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            "Pinned-image preflight failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(payload, dict):
        raise PackageContractError("Pinned-image probe returned no object.")
    return payload


def validate_package(
    *,
    deep: bool = False,
    image_path: Path | None = None,
    container_runtime: str | None = None,
    require_launch_ready: bool = False,
) -> dict[str, Any]:
    manifest = load_json(PACKAGE_DIR / "package_manifest.json", label="manifest")
    verify_self_digest(manifest, label="manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_matched_six_cell"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("target_horizon") != 50
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
    ):
        raise PackageContractError("Inert package identity drifted.")
    for key in (
        "bundle_manifest",
        "bundle_expected_artifacts",
        "bundle_source_locks",
        "bundle_validation_report",
        "source_archive_manifest",
        "execution_plan",
        "source_lock_audit",
        "scientific_equivalence",
    ):
        _binding(manifest.get(key), label=key, canonical=True)
    source_archive_path = _binding(
        manifest.get("source_archive"), label="source archive"
    )
    if (
        source_archive_path.stat().st_size != V3_SOURCE_ARCHIVE_SIZE_BYTES
        or sha256_file(source_archive_path) != V3_SOURCE_ARCHIVE_SHA256
    ):
        raise PackageContractError("Source archive lost exact sealed-v3 identity.")
    _binding(manifest.get("queue"), label="queue")
    jobs = manifest.get("jobs")
    protocols = manifest.get("protocols")
    if not isinstance(jobs, list) or not isinstance(protocols, list):
        raise PackageContractError("Job/protocol closure is absent.")
    if len(jobs) != 6 or len(protocols) != 6:
        raise PackageContractError("Job/protocol count drifted.")
    control_files = manifest.get("control_files")
    application_sources = manifest.get("application_source_contracts")
    if not isinstance(control_files, list) or len(control_files) != 8:
        raise PackageContractError("Control-file closure drifted.")
    if not isinstance(application_sources, list) or len(application_sources) != 3:
        raise PackageContractError("Application-source closure drifted.")
    for row in control_files:
        _binding(row, label="control file")
    for row in application_sources:
        _binding(row, label="application source contract", canonical=True)
    loaded_jobs: list[dict[str, Any]] = []
    for row in jobs:
        path = _binding(row, label="job", canonical=True)
        loaded_jobs.append(load_json(path, label="job"))
    for row in protocols:
        _binding(row, label="protocol", canonical=True)
    if any(
        row.get("method") not in RESOURCE_ENVELOPES
        or row.get("resources") != RESOURCE_ENVELOPES[row["method"]]
        for row in loaded_jobs
    ):
        raise PackageContractError("Right-sized v4 resource envelopes drifted.")
    equivalence = load_json(
        PACKAGE_DIR / SCIENTIFIC_EQUIVALENCE_RELATIVE,
        label="v3 scientific equivalence",
    )
    verify_self_digest(equivalence, label="v3 scientific equivalence")
    if (
        equivalence.get("status")
        != "passed_exact_source_and_scientific_semantics"
        or equivalence.get("v3_package_manifest_canonical_sha256")
        != V3_PACKAGE_MANIFEST_CANONICAL_SHA256
        or equivalence.get("implementation_source_inventory_sha256")
        != V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256
        or equivalence.get("source_archive", {}).get("sha256")
        != V3_SOURCE_ARCHIVE_SHA256
        or equivalence.get("source_archive", {}).get("exact_v3_byte_identity")
        is not True
        or equivalence.get("scientific_semantic_drift_detected") is not False
    ):
        raise PackageContractError("Sealed-v3 scientific equivalence drifted.")
    source_manifest = load_json(
        PACKAGE_DIR / "source/source_archive_manifest.json",
        label="source archive manifest",
    )
    member_paths = {
        str(row.get("path"))
        for row in source_manifest.get("members", [])
        if isinstance(row, Mapping)
    }
    if not set(REQUIRED_ROUTE_SOURCE_PATHS).issubset(member_paths):
        raise PackageContractError("Required route source archive closure drifted.")
    submit_template = (PACKAGE_DIR / "submit.sub.in").read_text(encoding="utf-8")
    if (
        f'+JobBatchName = "{BATCH_NAME}"' not in submit_template
        or f"={STAGING_OUTPUT_ROOT}/$(execution_id)" not in submit_template
    ):
        raise PackageContractError("Submit batch/staging identity drifted.")
    declared = {"package_manifest.json"}
    for key in (
        "bundle_manifest",
        "bundle_expected_artifacts",
        "bundle_source_locks",
        "bundle_validation_report",
        "source_archive",
        "source_archive_manifest",
        "queue",
        "execution_plan",
        "source_lock_audit",
        "scientific_equivalence",
    ):
        row = manifest.get(key)
        if isinstance(row, Mapping):
            declared.add(str(row.get("path", "")))
    for key in (
        "control_files",
        "application_source_contracts",
        "jobs",
        "protocols",
    ):
        rows = manifest.get(key)
        if isinstance(rows, list):
            declared.update(
                str(row.get("path", ""))
                for row in rows
                if isinstance(row, Mapping)
            )
    observed_files = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file()
    }
    undeclared = sorted(observed_files - declared)
    missing_declared = sorted(declared - observed_files)
    if undeclared or missing_declared:
        raise PackageContractError(
            "Exact package filesystem closure drifted: "
            f"undeclared={undeclared}, missing={missing_declared}."
        )

    # Deep validation executes every worker preflight in the pinned image.
    # Do not first import the locked source with the login-node interpreter:
    # CHTC currently exposes Python 3.9 there, while the execution image and
    # source contract use Python 3.10+ syntax.  The ordinary local/inert path
    # retains the inexpensive host preflights.
    shallow = (
        []
        if deep
        else [
            preflight(PACKAGE_DIR / "jobs" / f"{execution_id}.json")
            for execution_id in expected_ids
        ]
    )
    if any(row.get("status") != "passed" for row in shallow):
        raise PackageContractError("Worker preflight did not pass.")

    launch_ready = False
    pinned_probe: dict[str, Any] | None = None
    deep_count = 0
    if deep:
        if image_path is None or not image_path.is_file():
            raise PackageContractError("Deep validation requires the image.")
        if sha256_file(image_path) != REMOTE_IMAGE_SHA256:
            raise PackageContractError("Pinned image SHA-256 drifted.")
        runtime = _runtime(container_runtime)
        source_archive = PACKAGE_DIR / "source/source_locked.tar.gz"
        probe = _inside_image(
            runtime=runtime,
            image_path=image_path,
            arguments=[
                (PACKAGE_DIR / "probe_image_runtime.py").as_posix(),
                "--source-archive",
                source_archive.as_posix(),
            ],
        )
        if (
            probe.get("status") != "passed"
            or probe.get("resolved_backend_name") != "FakeMarrakesh"
            or probe.get("backend_resolution_kind") != "fake_exact"
        ):
            raise PackageContractError("Pinned image Qiskit probe drifted.")
        for execution_id in expected_ids:
            row = _inside_image(
                runtime=runtime,
                image_path=image_path,
                arguments=[
                    (PACKAGE_DIR / "run_cell.py").as_posix(),
                    "--preflight",
                    "--job",
                    (
                        PACKAGE_DIR
                        / "jobs"
                        / f"{execution_id}.json"
                    ).as_posix(),
                ],
            )
            if row.get("status") != "passed":
                raise PackageContractError("Deep worker preflight failed.")
            deep_count += 1
        pinned_probe = {
            "status": "passed",
            "image_sha256": REMOTE_IMAGE_SHA256,
            "probe": probe,
        }
        launch_ready = deep_count == 6
    if require_launch_ready and not launch_ready:
        raise PackageContractError("Launch-ready validation was required.")
    return digested(
        {
            "schema": "gradient_phase0_phase23_qiskit_package_validation_v1",
            "status": "passed_inert_package",
            "package_manifest_sha256": manifest["sha256"],
            "shallow_worker_preflight_count": len(shallow),
            "host_preflight_skipped_for_pinned_deep_validation": deep,
            "deep_worker_preflight_count": deep_count,
            "deep_worker_preflight_runtime": (
                "pinned_execution_image" if launch_ready else None
            ),
            "pinned_image_runtime_probe": pinned_probe,
            "launch_ready": launch_ready,
            "execution_authorized": False,
            "submission_authorized": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--container-runtime")
    parser.add_argument("--require-launch-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_package(
            deep=args.deep,
            image_path=args.image,
            container_runtime=args.container_runtime,
            require_launch_ready=args.require_launch_ready,
        )
    except (OSError, ValueError, PackageContractError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
