#!/usr/bin/env python3
"""Build the immutable two-row Geo projected-singleton packaging repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ID = "paper_i_hh_geo_projected_singleton_is_ss_r50_packaging_repair_20260719_v6_chtc"
PARENT_BUNDLE_ID = "paper_i_hh_geo_projected_singleton_all_six_r50_20260719_v5_chtc"
PARENT_CLUSTER_ID = 8887546
PARENT_PROCS = (4, 5)
SOURCE_SHA256 = "8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
JOB_IDS = (
    "geo_projected_singleton__intermediate_strong__r50",
    "geo_projected_singleton__strong_strong__r50",
)
MEMORY_MB = 65536
DISK_MB = 32768


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_exact(source: Path, destination: Path, expected_sha256: str | None = None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual = _sha256(destination)
    expected = expected_sha256 or _sha256(source)
    if actual != expected:
        raise RuntimeError(f"exact-copy SHA-256 mismatch for {destination}: {actual}!={expected}")
    return actual


def _write_submit(bundle: Path) -> None:
    relative = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    lines = [
        "universe = vanilla",
        "batch_name = paper-i-hh-geo-projected-singleton-is-ss-r50-packaging-repair-20260719-v6",
        f"executable = {relative}/execute_source_locked_job.sh",
        (
            f"arguments = $(job_manifest) {relative}/source_locked.tar.gz {SOURCE_SHA256} "
            f"chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(job_id)"
        ),
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "preserve_relative_paths = True",
        (
            f"transfer_input_files = {relative}/run_job.py, $(job_manifest), "
            f"$(normalized_manifest), {relative}/source_archive_manifest.json, "
            f"{relative}/bundle_manifest.json, {relative}/source_locked.tar.gz, "
            "chtc/phase3_optuna/image.sif"
        ),
        f"transfer_output_files = raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz",
        (
            f'transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz = '
            '$(job_id)_transfer.tar.gz"'
        ),
        "request_cpus = $(request_cpus)",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        "+WantFlocking = true",
        f"log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log",
        f"output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out",
        f"error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err",
        "requirements = TARGET.HasSIF",
        (
            "queue job_id, job_manifest, normalized_manifest, memory_mb, disk_mb, request_cpus "
            f"from {relative}/queue.tsv"
        ),
        "",
    ]
    (bundle / "submit.sub").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict[str, Any]:
    repo = _repo_root()
    bundle = Path(__file__).resolve().parent
    parent = repo / "chtc" / "phase3_optuna" / "input" / PARENT_BUNDLE_ID
    parent_hashes = json.loads(
        (parent / "submission_artifact_hashes.json").read_text(encoding="utf-8")
    )["files"]

    exact_copies = (
        "run_job.py",
        "remote_execution_gate.json",
        "source_archive_manifest.json",
        "source_locked.tar.gz",
        "settings_difference_audit.json",
        "visible_source_map_resolved.json",
    )
    copied_hashes: dict[str, str] = {}
    for relative in exact_copies:
        copied_hashes[relative] = _copy_exact(
            parent / relative,
            bundle / relative,
            parent_hashes[relative],
        )
    if copied_hashes["source_locked.tar.gz"] != SOURCE_SHA256:
        raise RuntimeError("source archive drifted from the parent v5 lock")
    gate = json.loads((bundle / "remote_execution_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "pass":
        raise RuntimeError("parent remote image gate is not passing")
    remote = gate.get("remote_execution_preflight", {})
    if remote.get("image_sha256") != IMAGE_SHA256:
        raise RuntimeError("remote image hash drifted from the parent v5 lock")

    for source in sorted((parent / "visible_source_locks").glob("*.json")):
        relative = f"visible_source_locks/{source.name}"
        copied_hashes[relative] = _copy_exact(
            source,
            bundle / relative,
            parent_hashes[relative],
        )

    queue_lines: list[str] = []
    records: list[dict[str, Any]] = []
    for job_id in JOB_IDS:
        job_relative = f"jobs/{job_id}.json"
        normalized_relative = f"normalized_manifests/{job_id}.json"
        job_sha = _copy_exact(
            parent / job_relative,
            bundle / job_relative,
            parent_hashes[job_relative],
        )
        normalized_sha = _copy_exact(
            parent / normalized_relative,
            bundle / normalized_relative,
            parent_hashes[normalized_relative],
        )
        if job_sha != normalized_sha:
            raise RuntimeError(f"job/normalized manifest mismatch for {job_id}")
        queue_lines.append(
            "\t".join(
                (
                    job_id,
                    f"chtc/phase3_optuna/input/{BUNDLE_ID}/{job_relative}",
                    f"chtc/phase3_optuna/input/{BUNDLE_ID}/{normalized_relative}",
                    str(MEMORY_MB),
                    str(DISK_MB),
                    "1",
                )
            )
        )
        job = json.loads((bundle / job_relative).read_text(encoding="utf-8"))
        records.append(
            {
                "job_id": job_id,
                "regime": job["regime"]["label"],
                "parent_job_manifest_sha256": job_sha,
                "parent_normalized_manifest_sha256": normalized_sha,
                "parent_bundle_id_recorded_in_manifest": job["bundle_id"],
            }
        )
    (bundle / "queue.tsv").write_text("\n".join(queue_lines) + "\n", encoding="utf-8")

    wrapper = bundle / "execute_source_locked_job.sh"
    wrapper.chmod(0o755)
    manifest = {
        "schema": "paper_i_hh_geo_projected_singleton_packaging_repair_bundle_v1",
        "bundle_id": BUNDLE_ID,
        "generated_utc": _utc_now(),
        "status": "prepared_not_submitted",
        "submission_enabled": True,
        "job_count": 2,
        "parent": {
            "bundle_id": PARENT_BUNDLE_ID,
            "cluster_id": PARENT_CLUSTER_ID,
            "held_procs_replaced": list(PARENT_PROCS),
            "hold_class": "missing_transfer_output_archive",
        },
        "records": records,
        "source_archive": {
            "path": f"chtc/phase3_optuna/input/{BUNDLE_ID}/source_locked.tar.gz",
            "sha256": SOURCE_SHA256,
        },
        "expected_image_sha256": IMAGE_SHA256,
        "scientific_settings_changed": [],
        "exact_parent_scientific_artifacts": {
            relative: copied_hashes[relative]
            for relative in sorted(copied_hashes)
        },
        "operational_changes": [
            "new immutable bundle and output identity",
            "queue narrowed to held parent procs 4 and 5 only",
            "EXIT, TERM, and INT fail-safe packaging of the job-owned diagnostic directory",
        ],
        "scientific_blockers": [],
        "operational_blockers": [],
    }
    _write_json(bundle / "bundle_manifest.json", manifest)
    _write_submit(bundle)
    hashes = {
        path.relative_to(bundle).as_posix(): _sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
        and path.name != "submission_artifact_hashes.json"
        and "__pycache__" not in path.parts
    }
    _write_json(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_geo_projected_singleton_packaging_repair_hashes_v1",
            "files": hashes,
        },
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("pass --build")
    print(json.dumps(build(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

