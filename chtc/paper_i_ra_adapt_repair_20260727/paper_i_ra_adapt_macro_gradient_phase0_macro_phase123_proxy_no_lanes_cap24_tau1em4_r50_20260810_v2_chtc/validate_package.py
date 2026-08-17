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
    ALGORITHM_ID,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REMOTE_IMAGE_SHA256,
    SOURCE_ACTIVATION_POLICY,
    STRUCTURAL_PROXY_MODE,
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
        or manifest.get("status") != "passed_inert_six_cells"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("weak_holstein_horizon") != 50
        or manifest.get("strong_holstein_horizon") != 50
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
    ):
        _binding(manifest.get(key), label=key, canonical=True)
    _binding(manifest.get("source_archive"), label="source archive")
    _binding(manifest.get("queue"), label="queue")
    jobs = manifest.get("jobs")
    protocols = manifest.get("protocols")
    if not isinstance(jobs, list) or not isinstance(protocols, list):
        raise PackageContractError("Job/protocol closure is absent.")
    if len(jobs) != 6 or len(protocols) != 6:
        raise PackageContractError("Job/protocol count drifted.")
    for row in jobs:
        _binding(row, label="job", canonical=True)
    for row in protocols:
        _binding(row, label="protocol", canonical=True)

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
            or probe.get("sealed_source_imported") is not True
            or probe.get("algorithm_id")
            != ALGORITHM_ID
            or probe.get("candidate_adapter_id") != CANDIDATE_ADAPTER_ID
            or probe.get("structural_proxy_mode") != STRUCTURAL_PROXY_MODE
            or probe.get("selector_qiskit_compile_cost_active") is not False
            or probe.get("source_activation_policy")
            != SOURCE_ACTIVATION_POLICY
            or probe.get("source_cwd_isolated") is not True
            or probe.get("sealed_module_paths_verified") is not True
            or int(probe.get("loaded_source_module_count", 0)) < 4
            or not isinstance(probe.get("sealed_module_paths"), Mapping)
        ):
            raise PackageContractError(
                "Pinned-image macro Phase-0 proxy probe drifted."
            )
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
            "schema": (
                "macro_gradient_phase0_proxy_no_lanes_package_validation_v2"
            ),
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
