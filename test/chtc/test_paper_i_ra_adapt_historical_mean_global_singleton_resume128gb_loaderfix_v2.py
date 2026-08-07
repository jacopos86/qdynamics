from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
OLD_PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc"
)
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_loaderfix_v2_chtc"
)
ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_loaderfix_v2_chtc_"
    "activation_ordinary_v1"
)
PACKAGE_MANIFEST_SHA256 = (
    "84d8f7bdcc79e986c8bbd22af8f3c1c5ed2d5c1b95aeb1e84affb5c3ae87e1a1"
)
ACTIVATION_MANIFEST_SHA256 = (
    "36bd7278293f4a32f010e1a6b733a35159ac8785a420113a3980276d4ee935c5"
)
ROUTE_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)
RESUME_LOADER_BEFORE_SHA256 = (
    "6d3753f22071cae21eb5eb006e634655be0fb4a9ec60054d61dfef2a3625e37f"
)
RESUME_LOADER_AFTER_SHA256 = (
    "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
)
IMPLEMENTATION_REPAIR_ID = (
    "accepted_round_current_checkpoint_receipt_loader_fix_v2"
)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _jobs(package: Path) -> list[dict]:
    manifest = _json(package / "package_manifest.json")
    return [_json(package / row["path"]) for row in manifest["jobs"]]


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }


def test_loaderfix_package_changes_only_bound_implementation_plumbing() -> None:
    old_jobs = _jobs(OLD_PACKAGE_DIR)
    jobs = _jobs(PACKAGE_DIR)
    manifest = _json(PACKAGE_DIR / "package_manifest.json")
    preserved_fields = (
        "source_execution_id",
        "regime_id",
        "nph",
        "route_id",
        "route_profile",
        "route_contract_sha256",
        "source_cluster_id",
        "source_proc_id",
        "target_horizon",
        "source_package",
        "source_job",
        "source_protocol",
        "source_archive_sha256",
        "source_runner_sha256",
        "resume_input",
        "resources",
        "scientific_protocol_sha256",
        "scientific_protocol_changed",
        "scientific_settings_changed",
    )

    assert manifest["sha256"] == PACKAGE_MANIFEST_SHA256
    assert manifest["scientific_protocol_changed"] is False
    assert manifest["scientific_settings_changed"] == []
    assert manifest["implementation_repair"] == {
        "repair_id": IMPLEMENTATION_REPAIR_ID,
        "path": "pipelines/static_adapt/sr_snake/_resume.py",
        "before_sha256": RESUME_LOADER_BEFORE_SHA256,
        "after_sha256": RESUME_LOADER_AFTER_SHA256,
        "scientific_protocol_changed": False,
        "scientific_settings_changed": [],
    }
    assert len(old_jobs) == len(jobs) == 3
    assert [job["resume_input"]["resume_controller_round"] for job in jobs] == [
        35,
        31,
        17,
    ]
    for old, job in zip(old_jobs, jobs, strict=True):
        assert all(old[field] == job[field] for field in preserved_fields)
        assert job["route_contract_sha256"] == ROUTE_CONTRACT_SHA256
        assert job["resources"]["request_memory_mb"] == 131_072
        assert job["resources"]["request_disk_mb"] == 81_920
        assert job["implementation_repair"] == manifest[
            "implementation_repair"
        ]
        assert job["execution_id"].endswith("_loaderfix_v2")


def test_loaderfix_source_preflight_imports_exact_repaired_bytes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "validate_package.py"),
            "--metadata-only",
            "--source-preflight",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["package_manifest_sha256"] == PACKAGE_MANIFEST_SHA256
    assert receipt["source_preflight_count"] == 3
    assert receipt["implementation_repair"]["repair_id"] == (
        IMPLEMENTATION_REPAIR_ID
    )
    assert receipt["implementation_repair"]["after_sha256"] == (
        RESUME_LOADER_AFTER_SHA256
    )
    active_loader = REPO_ROOT / "pipelines/static_adapt/sr_snake/_resume.py"
    assert hashlib.sha256(active_loader.read_bytes()).hexdigest() == (
        RESUME_LOADER_AFTER_SHA256
    )


def test_loaderfix_activation_is_ordinary_authorized_and_unsubmitted() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed_authorized_not_submitted"
    assert receipt["activation_manifest_sha256"] == (
        ACTIVATION_MANIFEST_SHA256
    )
    assert receipt["package_manifest_sha256"] == PACKAGE_MANIFEST_SHA256
    assert receipt["source_preflight_count"] == 3
    assert receipt["resume_controller_rounds"] == [35, 31, 17]
    assert receipt["implementation_repair"]["repair_id"] == (
        IMPLEMENTATION_REPAIR_ID
    )
    assert receipt["source_held_job_removal_authorized"] is False
    assert receipt["remote_stage"] is False
    assert receipt["condor_submit"] is False
    assert receipt["submitted"] is False

    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "periodic_release = False" in submit
    assert "leave_in_queue = False" in submit
    assert "hold = True" not in submit
    assert "max_materialize" not in submit.lower()


def test_attempt_archive_excludes_only_imported_resume_copy(
    tmp_path: Path,
) -> None:
    execution_id = "resume_failure_probe"
    worker = tmp_path / "worker_outputs"
    work = worker / f".{execution_id}.resume_work_v1"
    imported = work / "resume_input"
    artifacts = work / "artifacts"
    imported.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    imported_names = (
        "checkpoint.json",
        "checkpoint.estimator_call_ledger_checkpoint.old.json",
        "checkpoint.verified_singleton_resume.old.json",
    )
    for index, name in enumerate(imported_names, start=1):
        (imported / name).write_bytes(bytes([index]) * index)
    produced_names = (
        "checkpoint.json",
        "checkpoint.estimator_call_ledger_checkpoint.new.json",
        "checkpoint.verified_singleton_resume.new.json",
    )
    for name in produced_names:
        (artifacts / name).write_text(f"{name}\n", encoding="utf-8")
    (worker / "worker_exit_status.txt").write_text("137\n", encoding="utf-8")

    job = tmp_path / f"{execution_id}.json"
    authorization = tmp_path / "authorization.json"
    activation = tmp_path / "activation.json"
    for path in (job, authorization, activation):
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "transfer").mkdir()
    output = tmp_path / "transfer/attempt.tar.gz"
    digest = "a" * 64
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ACTIVATION_DIR / "build_attempt_archive.py"),
            "--worker-root",
            "worker_outputs",
            "--job",
            job.name,
            "--authorization",
            authorization.name,
            "--activation-manifest",
            activation.name,
            "--output-archive",
            "transfer/attempt.tar.gz",
            "--execution-id",
            execution_id,
            "--cluster-id",
            "1",
            "--proc-id",
            "0",
            "--attempt-ordinal",
            "1",
            "--worker-exit-status",
            "137",
            "--source-archive-sha256",
            digest,
            "--resume-archive-sha256",
            digest,
            "--image-sha256",
            digest,
        ],
        cwd=tmp_path,
        env=_environment(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["resumable_checkpoint_triplet_count"] == 1
    assert result["excluded_imported_resume_file_count"] == 3
    assert result["excluded_imported_resume_bytes"] == 6

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        receipt_stream = archive.extractfile("worker_attempt_receipt.json")
        assert receipt_stream is not None
        receipt = json.load(receipt_stream)
    imported_prefix = (
        f"worker_outputs/.{execution_id}.resume_work_v1/resume_input/"
    )
    produced_prefix = (
        f"worker_outputs/.{execution_id}.resume_work_v1/artifacts/"
    )
    assert not any(name.startswith(imported_prefix) for name in names)
    assert {produced_prefix + name for name in produced_names} <= names
    assert receipt["worker_exit_status"] == 137
    assert receipt["failure_safe_checkpoint_transfer"] is True
    assert receipt["resumable_checkpoint_triplet_count"] == 1
    assert receipt["imported_resume_input_excluded"] is True
    assert receipt["excluded_imported_resume_file_count"] == 3
    assert receipt["excluded_imported_resume_bytes"] == 6
    assert receipt["source_resume_archive_retained_separately"] is True
    assert not any(
        row["path"].startswith(
            f".{execution_id}.resume_work_v1/resume_input/"
        )
        for row in receipt["worker_files"]
    )
