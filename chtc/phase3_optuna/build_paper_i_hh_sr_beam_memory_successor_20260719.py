#!/usr/bin/env python3
"""Build a thin, operational-only memory successor for held beam-v4 rows.

The successor deliberately reuses the exact v4 worker, source archive, route
contract, and scientific job manifests.  Its only execution changes are the
Condor batch label and per-row memory requests recorded below.  It never
contacts or submits to CHTC.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "chtc/phase3_optuna/input"
PARENT_ID = (
    "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_"
    "symmetric_cost_noprune_no_ordinary_novelty_all_six_r50_20260718_v4_chtc"
)
SUCCESSOR_ID = (
    "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_"
    "symmetric_cost_noprune_no_ordinary_novelty_held4_memory_successor_"
    "r50_20260719_v5_chtc"
)
PARENT_BATCH = (
    "paper-i-hh-sr-appendix-historical-beam3x2-fullresp-symcost-noprune-"
    "nonovelty-six-r50-20260718-v4"
)
SUCCESSOR_BATCH = (
    "paper-i-hh-sr-appendix-historical-beam3x2-memory-repair-held4-"
    "r50-20260719-v5"
)
PARENT_CLUSTER = 8887576
SOURCE_ARCHIVE_SHA256 = (
    "77ef031ced6906718c8426ff703ec4c6c528495d956910a9a64a213d68432a04"
)
ROUTE_CONTRACT_SHA256 = (
    "49fb8c2f069722ce87cbaaedc8d7d32726a11dad92a624e3326269d75dcd1168"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
MEMORY_POLICY = "ceil_1p10_observed_memory_usage_to_4096mb_tier_v1"
BUILDER_SCRIPT_NAME = Path(__file__).name
RETRY_SCOPE = "held_parent_procs_0_1_2_5_only"
UNAFFECTED_PARENT_PROCS = [3, 4]

# Condor ClassAd evidence from the held v4 rows.  PeakMemoryUsage was absent;
# MemoryUsage and the raw resident set are preserved exactly as observed.
ROWS: tuple[dict[str, Any], ...] = (
    {
        "proc": 0,
        "slug": "weak_weak",
        "old_memory_mb": 32768,
        "observed_memory_usage_mb": 34180,
        "resident_set_size_raw": 33226964,
        "image_size": 35000000,
        "new_memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "proc": 1,
        "slug": "intermediate_weak",
        "old_memory_mb": 32768,
        "observed_memory_usage_mb": 34180,
        "resident_set_size_raw": 33200240,
        "image_size": 35000000,
        "new_memory_mb": 40960,
        "disk_mb": 61440,
    },
    {
        "proc": 2,
        "slug": "strong_weak_u8",
        "old_memory_mb": 40960,
        "observed_memory_usage_mb": 41504,
        "resident_set_size_raw": 41656140,
        "image_size": 42500000,
        "new_memory_mb": 49152,
        "disk_mb": 61440,
    },
    {
        "proc": 5,
        "slug": "strong_strong_u8",
        "old_memory_mb": 49152,
        "observed_memory_usage_mb": 48829,
        "resident_set_size_raw": 49963624,
        "image_size": 75000000,
        "new_memory_mb": 57344,
        "disk_mb": 81920,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def scientific_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove the two permitted operational deltas before equality checks."""
    value = copy.deepcopy(payload)
    value.pop("operational_retry", None)
    value.pop("batch_name", None)
    resource = value.get("resource_request")
    if isinstance(resource, dict):
        resource.pop("memory_mb", None)
    return value


def build() -> Path:
    parent = INPUT / PARENT_ID
    successor = INPUT / SUCCESSOR_ID
    if successor.exists():
        raise FileExistsError(f"immutable successor already exists: {successor}")
    if sha256(parent / "source_locked.tar.gz") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("parent source archive hash drift")

    successor.mkdir(parents=True)
    (successor / "jobs").mkdir()
    (successor / "normalized_manifests").mkdir()

    receipt_rows: list[dict[str, Any]] = []
    queue_lines: list[str] = []
    for row in ROWS:
        slug = row["slug"]
        parent_job = load(parent / "jobs" / f"{slug}.json")
        parent_norm = load(parent / "normalized_manifests" / f"{slug}.json")
        if parent_job["batch_name"] != PARENT_BATCH:
            raise ValueError(f"unexpected parent batch for {slug}")
        if parent_job["resource_request"]["memory_mb"] != row["old_memory_mb"]:
            raise ValueError(f"unexpected parent memory for {slug}")
        if (
            parent_job["route_identity"]["profile_contract_sha256"]
            != ROUTE_CONTRACT_SHA256
        ):
            raise ValueError(f"route digest drift for {slug}")

        retry = {
            "schema": "paper_i_hh_sr_condor_memory_retry_v1",
            "classification": "operational_only_no_scientific_change_v1",
            "parent_cluster": PARENT_CLUSTER,
            "parent_proc": row["proc"],
            "failure_cluster": row.get("failure_cluster", PARENT_CLUSTER),
            "failure_proc": row.get("failure_proc", row["proc"]),
            "failed_request_memory_mb": row.get(
                "failed_request_memory_mb", row["old_memory_mb"]
            ),
            "parent_bundle_id": PARENT_ID,
            "hold_reason": "cgroup memory limit exceeded",
            "peak_memory_usage_available": False,
            "memory_policy": MEMORY_POLICY,
            "old_request_memory_mb": row["old_memory_mb"],
            "observed_memory_usage_mb": row["observed_memory_usage_mb"],
            "resident_set_size_raw": row["resident_set_size_raw"],
            "image_size": row["image_size"],
            "new_request_memory_mb": row["new_memory_mb"],
            "scientific_settings_changed": False,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        }
        for payload in (parent_job, parent_norm):
            payload["batch_name"] = SUCCESSOR_BATCH
            payload["resource_request"]["memory_mb"] = row["new_memory_mb"]
            payload["operational_retry"] = retry

        if scientific_view(parent_job) != scientific_view(
            load(parent / "jobs" / f"{slug}.json")
        ):
            raise AssertionError(f"scientific job drift for {slug}")
        if scientific_view(parent_norm) != scientific_view(
            load(parent / "normalized_manifests" / f"{slug}.json")
        ):
            raise AssertionError(f"normalized scientific drift for {slug}")

        job_path = successor / "jobs" / f"{slug}.json"
        norm_path = successor / "normalized_manifests" / f"{slug}.json"
        dump(job_path, parent_job)
        dump(norm_path, parent_norm)
        queue_lines.append(
            "\t".join(
                (
                    slug,
                    str(job_path.relative_to(REPO)),
                    str(norm_path.relative_to(REPO)),
                    str(row["new_memory_mb"]),
                    str(row["disk_mb"]),
                )
            )
        )
        receipt_rows.append(retry)

    (successor / "queue.tsv").write_text(
        "\n".join(queue_lines) + "\n", encoding="utf-8"
    )

    parent_rel = f"chtc/phase3_optuna/input/{PARENT_ID}"
    successor_rel = f"chtc/phase3_optuna/input/{SUCCESSOR_ID}"
    parent_inputs = [
        "run_job.py",
        "evidence_validation.py",
        "validate_fetched.py",
        "source_archive_manifest.json",
        "source_revision_manifest.json",
        "physics_and_exact_reference_lock.json",
        "bundle_manifest.json",
        "preflight.json",
        "route_parity.json",
        "scientific_settings_audit.json",
    ]
    transfer_inputs = [f"{parent_rel}/{name}" for name in parent_inputs]
    transfer_inputs += ["$(job_manifest)", "$(normalized_manifest)"]
    transfer_inputs += [f"{parent_rel}/source_locked.tar.gz", "chtc/phase3_optuna/image.sif"]
    submit = f'''universe = vanilla
# Operational-only successor for beam-v4 rows held by cgroup memory.
# Scientific source, route digest, commands, horizons, and output semantics are unchanged.
executable = {parent_rel}/execute_source_locked_job.sh
arguments = $(job_manifest) {parent_rel}/source_locked.tar.gz {SOURCE_ARCHIVE_SHA256} chtc/phase3_optuna/image.sif {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {', '.join(transfer_inputs)}
transfer_output_files = raw_outputs/{PARENT_ID}/$(regime_slug)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{PARENT_ID}/$(regime_slug)_transfer.tar.gz = $(Cluster).$(Process)__$(regime_slug)_transfer.tar.gz"
stream_output = False
stream_error = False
log = logs/{SUCCESSOR_ID}.$(Cluster).$(Process).log
output = logs/{SUCCESSOR_ID}.$(Cluster).$(Process).out
error = logs/{SUCCESSOR_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{SUCCESSOR_BATCH}"
notification = Never
queue regime_slug, job_manifest, normalized_manifest, memory_mb, disk_mb from {successor_rel}/queue.tsv
'''
    (successor / "submit.sub").write_text(submit, encoding="utf-8")

    parent_artifacts = {
        name: {
            "path": f"{parent_rel}/{name}",
            "sha256": sha256(parent / name),
            "size_bytes": (parent / name).stat().st_size,
        }
        for name in ["execute_source_locked_job.sh", *parent_inputs, "source_locked.tar.gz"]
    }
    receipt = {
        "schema": "paper_i_hh_sr_beam_memory_successor_receipt_v1",
        "classification": "operational_only_no_scientific_change_v1",
        "parent_cluster": PARENT_CLUSTER,
        "parent_bundle_id": PARENT_ID,
        "parent_batch_name": PARENT_BATCH,
        "successor_bundle_id": SUCCESSOR_ID,
        "successor_batch_name": SUCCESSOR_BATCH,
        "submission_performed": False,
        "source_archive_reused_byte_identical": True,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "route_contract_unchanged": True,
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "scientific_settings_changed": False,
        "retry_scope": RETRY_SCOPE,
        "unaffected_parent_procs_not_duplicated": UNAFFECTED_PARENT_PROCS,
        "memory_policy": MEMORY_POLICY,
        "rows": receipt_rows,
        "quota_note": (
            "thin overlay; no duplicate source archive. Remote submission remains "
            "blocked until the parent artifacts are verified present and the reported "
            "submit-host quota headroom has been verified."
        ),
        "parent_artifacts": parent_artifacts,
    }
    dump(successor / "operational_memory_successor_receipt.json", receipt)

    preflight = {
        "schema": "paper_i_hh_sr_beam_memory_successor_preflight_v1",
        "status": "pass_local_not_submitted",
        "successor_bundle_id": SUCCESSOR_ID,
        "successor_batch_name": SUCCESSOR_BATCH,
        "parent_cluster": PARENT_CLUSTER,
        "parent_bundle_id": PARENT_ID,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "scientific_settings_changed": False,
        "queued_regimes": [row["slug"] for row in ROWS],
        "checks": {
            "retry_row_count_exact": len(receipt_rows) == len(ROWS),
            "unaffected_parent_procs_not_duplicated": all(
                proc not in {int(row["proc"]) for row in ROWS}
                for proc in UNAFFECTED_PARENT_PROCS
            ),
            "parent_source_archive_hash_exact": True,
            "route_digest_exact_all_rows": True,
            "scientific_manifest_views_byte_equivalent": True,
            "memory_policy_applied_exactly": True,
            "source_archive_not_duplicated": True,
            "submission_not_performed": True,
        },
        "remote_gates_pending": [
            "parent_artifact_hashes_on_submit_host",
            "container_image_hash_and_import_preflight",
            "home_quota_below_hard_limit",
            "condor_submit_retry_row_acceptance",
        ],
    }
    dump(successor / "preflight.json", preflight)
    dump(successor / "parent_artifact_requirements.json", parent_artifacts)

    readme = f"""# Beam-v4 held-row memory successor

This thin immutable overlay retries only parent cluster `{PARENT_CLUSTER}`
processes `0,1,2,5`, which were held by cgroup memory limits.  It reuses the
byte-identical v4 source archive, worker, route contract, scientific commands,
50-round horizons, and output validation.  Processes `3,4` are not duplicated.

Memory requests use `{MEMORY_POLICY}` and are recorded per row in
`operational_memory_successor_receipt.json`.  The builder does not upload or
submit anything.
"""
    (successor / "README.md").write_text(readme, encoding="utf-8")

    verifier = f'''#!/usr/bin/env python3
"""Verify the immutable operational-only beam memory successor."""
import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[2] / {BUILDER_SCRIPT_NAME!r}
spec = importlib.util.spec_from_file_location("beam_memory_builder", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
if __name__ == "__main__":
    module.verify(Path(__file__).resolve().parent)
    print("beam memory successor verification passed")
'''
    (successor / "build_bundle.py").write_text(verifier, encoding="utf-8")
    test_text = '''#!/usr/bin/env python3
import unittest
import build_bundle
class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        build_bundle.module.verify(build_bundle.Path(__file__).resolve().parent)
if __name__ == "__main__": unittest.main()
'''
    (successor / "test_bundle.py").write_text(test_text, encoding="utf-8")

    upload_paths = [
        f"{successor_rel}/{name}"
        for name in (
            "submit.sub",
            "queue.tsv",
            "operational_memory_successor_receipt.json",
            "preflight.json",
            "parent_artifact_requirements.json",
            "build_bundle.py",
            "test_bundle.py",
            "README.md",
            "submission_artifact_hashes.json",
        )
    ]
    upload_paths += [
        f"{successor_rel}/jobs/{row['slug']}.json" for row in ROWS
    ]
    upload_paths += [
        f"{successor_rel}/normalized_manifests/{row['slug']}.json" for row in ROWS
    ]
    (successor / "upload_artifact_list.txt").write_text(
        "\n".join(upload_paths) + "\n", encoding="utf-8"
    )

    verify(successor)
    hashes = {
        str(path.relative_to(successor)): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(successor.rglob("*"))
        if path.is_file() and path.name != "submission_artifact_hashes.json"
    }
    dump(
        successor / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_hh_sr_beam_memory_successor_hashes_v1",
            "successor_bundle_id": SUCCESSOR_ID,
            "artifacts": hashes,
        },
    )
    return successor


def verify(successor: Path | None = None) -> None:
    parent = INPUT / PARENT_ID
    successor = successor or INPUT / SUCCESSOR_ID
    if sha256(parent / "source_locked.tar.gz") != SOURCE_ARCHIVE_SHA256:
        raise AssertionError("parent source archive hash drift")
    queue = [
        line.split("\t")
        for line in (successor / "queue.tsv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [parts[0] for parts in queue] != [row["slug"] for row in ROWS]:
        raise AssertionError("retry queue does not contain exactly the held rows")
    receipt = load(successor / "operational_memory_successor_receipt.json")
    for requirement in receipt["parent_artifacts"].values():
        path = REPO / requirement["path"]
        if sha256(path) != requirement["sha256"]:
            raise AssertionError(f"parent artifact hash drift: {path}")
    submit = (successor / "submit.sub").read_text(encoding="utf-8")
    if f"{PARENT_ID}/source_locked.tar.gz" not in submit:
        raise AssertionError("submit does not reuse the parent source archive")
    if f'JobBatchName = "{SUCCESSOR_BATCH}"' not in submit:
        raise AssertionError("successor batch name missing")
    for row, parts in zip(ROWS, queue, strict=True):
        policy_memory = math.ceil(
            1.10 * row["observed_memory_usage_mb"] / 4096
        ) * 4096
        if policy_memory != row["new_memory_mb"]:
            raise AssertionError(f"memory policy drift for {row['slug']}")
        if int(parts[3]) != row["new_memory_mb"] or int(parts[4]) != row["disk_mb"]:
            raise AssertionError(f"queue resource drift for {row['slug']}")
        for folder in ("jobs", "normalized_manifests"):
            old = load(parent / folder / f"{row['slug']}.json")
            new = load(successor / folder / f"{row['slug']}.json")
            if scientific_view(old) != scientific_view(new):
                raise AssertionError(f"scientific drift in {folder}/{row['slug']}")
            if new["resource_request"]["memory_mb"] != row["new_memory_mb"]:
                raise AssertionError(f"manifest memory drift for {row['slug']}")
            if new["bundle_id"] != PARENT_ID:
                raise AssertionError("worker-compatible scientific bundle id changed")
        job = load(successor / "jobs" / f"{row['slug']}.json")
        if job["route_identity"]["profile_contract_sha256"] != ROUTE_CONTRACT_SHA256:
            raise AssertionError(f"route digest drift for {row['slug']}")


if __name__ == "__main__":
    path = build()
    print(path.relative_to(REPO))
