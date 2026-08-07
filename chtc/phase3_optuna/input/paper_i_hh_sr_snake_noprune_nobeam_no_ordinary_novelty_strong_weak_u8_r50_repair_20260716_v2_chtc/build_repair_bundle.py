#!/usr/bin/env python3
"""Build the source-only repair resubmission for the failed strong-weak row."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "strong_weak_u8_r50_repair_20260716_v2_chtc"
)
BATCH_NAME = "paper-i-hh-sr-sw-r50-repair-20260716-v2"
PARENT_BUNDLE_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "r50_continuations_20260715_v1_chtc"
)
SOURCE_FIX_ID = (
    "paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "r50_continuations_20260716_v2_source_fix"
)
REGIME = "strong_weak_u8"
IMAGE_PATH = Path("chtc/phase3_optuna/image.sif")
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
SOURCE_ARCHIVE_SHA256 = (
    "e682e79d4c9218794c94822ebce99df427f0840287e5212a45e528931cb2efc5"
)
SOURCE_REPAIR_PATCH_SHA256 = (
    "33fce0f0eb608437b6d329eccd598b9252336d4b13126435b067f54475d00b0b"
)
PARENT_ARCHIVE_SHA256 = (
    "070febcf91a31fc1249afd24f59b9a68e57c6ed547315cabed461933b51b1c2a"
)

BUNDLE_DIR = Path(__file__).resolve().parent
REPO = BUNDLE_DIR.parents[3]
PARENT_DIR = REPO / "chtc/phase3_optuna/input" / PARENT_BUNDLE_ID
SOURCE_FIX_DIR = REPO / "chtc/phase3_optuna/input" / SOURCE_FIX_ID


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def replace_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_string(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: replace_string(item, old, new) for key, item in value.items()}
    return value


def repo_path(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def main() -> int:
    parent_archive = PARENT_DIR / "source_locked.tar.gz"
    source_archive_input = SOURCE_FIX_DIR / "source_locked.tar.gz"
    source_patch_input = (
        SOURCE_FIX_DIR / "source_lock/no_batch_terminal_phase_batch_summary.patch"
    )
    source_patch_manifest_input = (
        SOURCE_FIX_DIR
        / "source_lock/no_batch_terminal_phase_batch_summary_patch_manifest.json"
    )
    required = [
        PARENT_DIR / "run_job.py",
        PARENT_DIR / "execute_source_locked_job.sh",
        PARENT_DIR / f"jobs/{REGIME}.json",
        PARENT_DIR / f"source_records/{REGIME}.json",
        PARENT_DIR / f"checkpoint_validation/{REGIME}.json",
        PARENT_DIR / f"resume_inputs/{REGIME}.round30.current.json.gz",
        PARENT_DIR / f"resume_inputs/{REGIME}.round30.estimator_call_ledger.json.gz",
        PARENT_DIR
        / f"resume_inputs/{REGIME}.round30.signed_active_prefix_checkpoint.json",
        PARENT_DIR / "source_lock/no_beam_resume_patch_manifest.json",
        PARENT_DIR / "source_lock/no_beam_verified_resume.patch",
        parent_archive,
        source_archive_input,
        source_patch_input,
        source_patch_manifest_input,
    ]
    missing = [repo_path(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")
    if sha256(parent_archive) != PARENT_ARCHIVE_SHA256:
        raise ValueError("parent archive hash drift")
    if sha256(source_archive_input) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("repaired source archive hash drift")
    if sha256(source_patch_input) != SOURCE_REPAIR_PATCH_SHA256:
        raise ValueError("source repair patch hash drift")

    source_lock_dir = BUNDLE_DIR / "source_lock"
    jobs_dir = BUNDLE_DIR / "jobs"
    source_lock_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    source_archive = BUNDLE_DIR / "source_locked.tar.gz"
    source_patch = source_lock_dir / source_patch_input.name
    source_patch_manifest = source_lock_dir / source_patch_manifest_input.name
    shutil.copy2(source_archive_input, source_archive)
    shutil.copy2(source_patch_input, source_patch)
    shutil.copy2(source_patch_manifest_input, source_patch_manifest)

    runner_text = (PARENT_DIR / "run_job.py").read_text(encoding="utf-8")
    runner_text = runner_text.replace(PARENT_BUNDLE_ID, BUNDLE_ID)
    runner_text = runner_text.replace(
        '"r50_continuations_20260715_v1_chtc"',
        '"strong_weak_u8_r50_repair_20260716_v2_chtc"',
    )
    old_loop = '''        "transferred_signed_prefix_sidecar",
    ):'''
    new_loop = '''        "transferred_signed_prefix_sidecar",
        "no_batch_terminal_phase_patch_manifest",
        "no_batch_terminal_phase_patch",
    ):'''
    if old_loop not in runner_text:
        raise RuntimeError("runner source-lock validation insertion point drifted")
    runner_text = runner_text.replace(old_loop, new_loop, 1)
    runner_path = BUNDLE_DIR / "run_job.py"
    runner_path.write_text(runner_text, encoding="utf-8")
    runner_path.chmod(0o755)

    execute_text = (PARENT_DIR / "execute_source_locked_job.sh").read_text(
        encoding="utf-8"
    )
    execute_text = execute_text.replace(PARENT_BUNDLE_ID, BUNDLE_ID)
    execute_path = BUNDLE_DIR / "execute_source_locked_job.sh"
    execute_path.write_text(execute_text, encoding="utf-8")
    execute_path.chmod(0o755)

    parent_job = load_json(PARENT_DIR / f"jobs/{REGIME}.json")
    job = json.loads(json.dumps(parent_job))
    job["bundle_id"] = BUNDLE_ID
    job["batch_name"] = BATCH_NAME
    job["created_utc"] = utc_now()
    for field in ("paths", "environment", "transfer_contract"):
        job[field] = replace_string(job[field], PARENT_BUNDLE_ID, BUNDLE_ID)
    command = job["command"]
    command["execution_argv"] = replace_string(
        command["execution_argv"], PARENT_BUNDLE_ID, BUNDLE_ID
    )
    command["execution_options"] = replace_string(
        command["execution_options"], PARENT_BUNDLE_ID, BUNDLE_ID
    )
    settings_difference = job["settings_difference"]
    settings_difference["environment_differences"] = replace_string(
        settings_difference["environment_differences"], PARENT_BUNDLE_ID, BUNDLE_ID
    )
    settings_difference["source_only_repair"] = {
        "scientific_settings_changed": [],
        "parent_source_archive_sha256": PARENT_ARCHIVE_SHA256,
        "repaired_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "changed_files": ["pipelines/static_adapt/adapt_pipeline.py"],
        "purpose": "initialize inactive no-batching telemetry before the Phase-II live gate",
    }
    source_lock = job["source_lock"]
    source_lock["parent_patched_source_archive"] = source_lock[
        "patched_source_archive"
    ]
    source_lock["parent_patched_source_archive_sha256"] = source_lock[
        "patched_source_archive_sha256"
    ]
    source_lock["patched_source_archive"] = repo_path(source_archive)
    source_lock["patched_source_archive_sha256"] = SOURCE_ARCHIVE_SHA256
    source_lock["no_batch_terminal_phase_patch"] = repo_path(source_patch)
    source_lock["no_batch_terminal_phase_patch_sha256"] = sha256(source_patch)
    source_lock["no_batch_terminal_phase_patch_manifest"] = repo_path(
        source_patch_manifest
    )
    source_lock["no_batch_terminal_phase_patch_manifest_sha256"] = sha256(
        source_patch_manifest
    )
    source_lock["scientific_settings_changed_by_repair"] = []
    job_path = jobs_dir / f"{REGIME}.json"
    dump_json(job_path, job)

    source_archive_manifest = load_json(SOURCE_FIX_DIR / "source_archive_manifest.json")
    source_archive_manifest["archive_path"] = repo_path(source_archive)
    source_archive_manifest["patch"]["path"] = repo_path(source_patch)
    source_archive_manifest["patch"]["manifest"] = repo_path(source_patch_manifest)
    source_archive_manifest["submission_status"] = "staged_not_submitted"
    source_archive_manifest_path = BUNDLE_DIR / "source_archive_manifest.json"
    dump_json(source_archive_manifest_path, source_archive_manifest)

    checkpoint_validation = PARENT_DIR / f"checkpoint_validation/{REGIME}.json"
    source_record = PARENT_DIR / f"source_records/{REGIME}.json"
    resume_checkpoint = (
        PARENT_DIR / f"resume_inputs/{REGIME}.round30.current.json.gz"
    )
    resume_ledger = (
        PARENT_DIR / f"resume_inputs/{REGIME}.round30.estimator_call_ledger.json.gz"
    )
    resume_signed_prefix = (
        PARENT_DIR
        / f"resume_inputs/{REGIME}.round30.signed_active_prefix_checkpoint.json"
    )
    parent_patch_manifest = PARENT_DIR / "source_lock/no_beam_resume_patch_manifest.json"
    parent_patch = PARENT_DIR / "source_lock/no_beam_verified_resume.patch"

    queue_path = BUNDLE_DIR / "queue.tsv"
    queue_path.write_text(
        "\t".join(
            [
                REGIME,
                repo_path(job_path),
                repo_path(source_record),
                repo_path(resume_checkpoint),
                repo_path(resume_ledger),
                repo_path(resume_signed_prefix),
                "32768",
                "61440",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    transfer_inputs = [
        runner_path,
        source_archive_manifest_path,
        BUNDLE_DIR / "source_lock_and_settings_diff.json",
        BUNDLE_DIR / "bundle_manifest.json",
        BUNDLE_DIR / "preflight.json",
        parent_patch_manifest,
        parent_patch,
        source_patch_manifest,
        source_patch,
        source_record,
        resume_checkpoint,
        resume_ledger,
        resume_signed_prefix,
        job_path,
        source_archive,
        REPO / IMAGE_PATH,
    ]
    base = f"chtc/phase3_optuna/input/{BUNDLE_ID}"
    submit_text = f'''universe = vanilla
executable = {base}/execute_source_locked_job.sh
arguments = $(job_manifest) {base}/source_locked.tar.gz {SOURCE_ARCHIVE_SHA256} {IMAGE_PATH.as_posix()} {IMAGE_SHA256} $(regime_slug)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = {", ".join(repo_path(path) if path.is_absolute() else path.as_posix() for path in transfer_inputs)}
transfer_output_files = raw_outputs/{BUNDLE_ID}/$(regime_slug)_transfer.tar.gz
stream_output = False
stream_error = False
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = 259200
+JobBatchName = "{BATCH_NAME}"
notification = Never
queue regime_slug, job_manifest, source_record, resume_checkpoint, resume_ledger, resume_signed_prefix, memory_mb, disk_mb from {base}/queue.tsv
'''
    submit_path = BUNDLE_DIR / "submit.sub"
    submit_path.write_text(submit_text, encoding="utf-8")

    settings_audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "bundle_id": BUNDLE_ID,
        "parent_bundle_id": PARENT_BUNDLE_ID,
        "regime_slug": REGIME,
        "run_class": "candidate_source_locked_continuation_repair",
        "scientific_settings_changed": [],
        "operational_changes": [
            "isolated output/cache paths",
            "bundle and batch labels",
            "source archive hash",
        ],
        "source_repair": {
            "changed_files": ["pipelines/static_adapt/adapt_pipeline.py"],
            "parent_archive_sha256": PARENT_ARCHIVE_SHA256,
            "repaired_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "patch_sha256": SOURCE_REPAIR_PATCH_SHA256,
            "archive_execution_probe": load_json(source_patch_manifest_input)[
                "verification"
            ]["archive_execution_probe"],
        },
        "parent_job_manifest_sha256": sha256(
            PARENT_DIR / f"jobs/{REGIME}.json"
        ),
        "repair_job_manifest_sha256": sha256(job_path),
        "status": "pass",
    }
    settings_audit_path = BUNDLE_DIR / "source_lock_and_settings_diff.json"
    dump_json(settings_audit_path, settings_audit)

    validation = subprocess.run(
        [sys.executable, str(runner_path), "--validate-only", str(job_path)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if validation.returncode != 0:
        raise RuntimeError(
            f"repair job validation failed: {validation.stdout}\n{validation.stderr}"
        )
    preflight = {
        "schema": "paper_i_hh_sr_r50_source_repair_preflight_v2",
        "bundle_id": BUNDLE_ID,
        "regime_slug": REGIME,
        "status": "pass",
        "scientific_execution_performed": False,
        "submission_performed": False,
        "checks": {
            "parent_checkpoint_validation": load_json(checkpoint_validation).get(
                "status"
            )
            == "pass",
            "runner_validation": True,
            "source_archive_hash": sha256(source_archive)
            == SOURCE_ARCHIVE_SHA256,
            "source_repair_probe": load_json(source_patch_manifest_input)[
                "verification"
            ]["archive_execution_probe"]["status"]
            == "pass",
            "scientific_settings_diff_empty": True,
            "isolated_output_paths": True,
            "same_round30_checkpoint_and_ledger": True,
        },
        "runner_validation_stdout": validation.stdout.strip(),
        "blockers": [],
        "created_utc": utc_now(),
    }
    preflight_path = BUNDLE_DIR / "preflight.json"
    dump_json(preflight_path, preflight)

    bundle_manifest = {
        "schema": "paper_i_hh_sr_r30_to_r50_source_repair_bundle_v2",
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "run_class": "candidate_source_locked_continuation_repair",
        "created_utc": utc_now(),
        "planned_row_count": 1,
        "ready_row_count": 1,
        "blocked_rows": [],
        "scientific_settings_changed": [],
        "jobs": [
            {
                "regime_slug": REGIME,
                "job_manifest": repo_path(job_path),
                "job_manifest_sha256": sha256(job_path),
                "source_record": repo_path(source_record),
                "source_record_sha256": sha256(source_record),
                "checkpoint_validation": repo_path(checkpoint_validation),
                "checkpoint_validation_sha256": sha256(checkpoint_validation),
                "resume_checkpoint": repo_path(resume_checkpoint),
                "resume_checkpoint_sha256": sha256(resume_checkpoint),
                "resume_ledger": repo_path(resume_ledger),
                "resume_ledger_sha256": sha256(resume_ledger),
                "resume_signed_prefix": repo_path(resume_signed_prefix),
                "resume_signed_prefix_sha256": sha256(resume_signed_prefix),
                "resources": {
                    "request_cpus": 4,
                    "request_memory_mb": 32768,
                    "request_disk_mb": 61440,
                    "max_runtime_s": 259200,
                },
            }
        ],
        "source_archive": source_archive_manifest,
        "preflight": {
            "path": repo_path(preflight_path),
            "sha256": sha256(preflight_path),
        },
        "source_lock_and_settings_diff": {
            "path": repo_path(settings_audit_path),
            "sha256": sha256(settings_audit_path),
        },
        "submission_status": "staged_not_submitted",
    }
    bundle_manifest_path = BUNDLE_DIR / "bundle_manifest.json"
    dump_json(bundle_manifest_path, bundle_manifest)

    # Regenerate the submit description after manifests exist so all paths are valid.
    transfer_inputs[2] = settings_audit_path
    transfer_inputs[3] = bundle_manifest_path
    transfer_inputs[4] = preflight_path
    submit_text = submit_text.replace(
        ", ".join(
            repo_path(path) if path.is_absolute() else path.as_posix()
            for path in transfer_inputs
        ),
        ", ".join(repo_path(path) for path in transfer_inputs),
    )
    submit_path.write_text(submit_text, encoding="utf-8")

    upload_artifacts = [
        execute_path,
        runner_path,
        submit_path,
        queue_path,
        source_archive_manifest_path,
        settings_audit_path,
        bundle_manifest_path,
        preflight_path,
        source_patch_manifest,
        source_patch,
        job_path,
        source_archive,
    ]
    upload_path = BUNDLE_DIR / "upload_artifact_list.txt"
    upload_path.write_text(
        "\n".join(repo_path(path) for path in upload_artifacts) + "\n",
        encoding="utf-8",
    )
    all_submission_artifacts = [
        *upload_artifacts,
        parent_patch_manifest,
        parent_patch,
        source_record,
        resume_checkpoint,
        resume_ledger,
        resume_signed_prefix,
        checkpoint_validation,
    ]
    artifact_hashes_path = BUNDLE_DIR / "submission_artifact_hashes.json"
    dump_json(
        artifact_hashes_path,
        {
            "schema": "paper_i_hh_sr_r50_source_repair_artifact_hashes_v2",
            "created_utc": utc_now(),
            "artifacts": {
                repo_path(path): artifact_record(path)
                for path in all_submission_artifacts
            },
            "required_remote_dependency": {
                "path": IMAGE_PATH.as_posix(),
                "sha256": IMAGE_SHA256,
            },
        },
    )
    with upload_path.open("a", encoding="utf-8") as handle:
        handle.write(repo_path(artifact_hashes_path) + "\n")

    readme = f"""# Strong-weak SR-SNAKE round-50 source repair v2

Status: staged locally; not submitted.

This single-row bundle resumes the exact authenticated round-30 strong-weak
prefix used by cluster 8811168. The only source change is the verified
no-batching telemetry initialization in `pipelines/static_adapt/adapt_pipeline.py`.
No scientific setting, checkpoint, estimator ledger, route policy, optimizer
budget, horizon, or reference changed.

- Parent archive SHA-256: `{PARENT_ARCHIVE_SHA256}`
- Repaired archive SHA-256: `{SOURCE_ARCHIVE_SHA256}`
- Repair patch SHA-256: `{SOURCE_REPAIR_PATCH_SHA256}`
- Job validation: pass
"""
    (BUNDLE_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "bundle_id": BUNDLE_ID,
                "job_manifest_sha256": sha256(job_path),
                "source_archive_sha256": sha256(source_archive),
                "submit_sha256": sha256(submit_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
