from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc"
)
ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v3_resume128gb_chtc_activation_ordinary_v1"
)
PACKAGE_MANIFEST_SHA256 = (
    "f34dfa4e7157ef6e009c5a78547c116989392a46d31d4c0448dc1fc87d7968b0"
)
ACTIVATION_MANIFEST_SHA256 = (
    "5fa7899f3fbc7ef7e5878216e70a5e00bbfd827f3f939ed0689de7c943dcabdf"
)
ROUTE_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _jobs() -> list[dict]:
    manifest = _json(PACKAGE_DIR / "package_manifest.json")
    return [_json(PACKAGE_DIR / row["path"]) for row in manifest["jobs"]]


def test_resume_package_preserves_exact_scientific_protocols() -> None:
    manifest = _json(PACKAGE_DIR / "package_manifest.json")
    jobs = _jobs()

    assert manifest["sha256"] == PACKAGE_MANIFEST_SHA256
    assert manifest["row_count"] == 3
    assert manifest["scientific_protocol_changed"] is False
    assert manifest["scientific_settings_changed"] == []
    assert manifest["source_held_jobs_preserved"] is True
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert [job["resume_input"]["resume_controller_round"] for job in jobs] == [
        35,
        31,
        17,
    ]
    for job in jobs:
        protocol = _json(REPO_ROOT / job["source_protocol"]["path"])
        assert protocol["sha256"] == job["scientific_protocol_sha256"]
        assert protocol["request"]["execution"]["resume"]["kind"] == (
            "fresh_start"
        )
        assert protocol["route_contract"]["sha256"] == (
            ROUTE_CONTRACT_SHA256
        )
        assert job["route_contract_sha256"] == ROUTE_CONTRACT_SHA256
        assert job["resources"]["request_memory_mb"] == 131_072
        assert job["resources"]["request_disk_mb"] == 81_920
        assert job["source_job_preserved_held"] is True
        resume = job["resume_input"]
        assert resume["pointer_closed"] is True
        assert resume["member_count"] == 3
        assert {row["role"] for row in resume["members"]} == {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }


def test_activation_validator_passes_exact_authorized_unsubmitted_closure() -> None:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "STATIC_ADAPT_HH_POOL_CACHE": "off",
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
    }
    completed = subprocess.run(
        [sys.executable, "-B", str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        env=environment,
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
    assert receipt["row_specific_resume_archives"] is True
    assert receipt["request_memory_mb"] == 131_072
    assert receipt["request_disk_mb"] == 81_920
    assert receipt["source_held_job_removal_authorized"] is False
    assert receipt["submitted"] is False
    assert not list(PACKAGE_DIR.rglob("__pycache__"))
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))


def test_submit_is_ordinary_unheld_and_transfers_one_archive_per_row() -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    queue = (ACTIVATION_DIR / "queue.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    wrapper = (ACTIVATION_DIR / "execute_resume_job.sh").read_text(
        encoding="utf-8"
    )

    assert len(queue) == 3
    assert all(len(row.split("\t")) == 11 for row in queue)
    assert len({row.split("\t")[5] for row in queue}) == 3
    assert all(row.split("\t")[8:10] == ["131072", "81920"] for row in queue)
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in submit
    assert "$(resume_archive_path)" in submit
    assert "request_memory = $(memory_mb)MB" in submit
    assert "request_disk = $(disk_mb)MB" in submit
    assert "periodic_release = False" in submit
    assert "leave_in_queue = False" in submit
    assert "kill_sig = SIGTERM" in submit
    assert "kill_sig_timeout = 600" in submit
    assert "max_materialize" not in submit.lower()
    assert "hold = True" not in submit
    assert 'trap package_attempt EXIT' in wrapper
    assert 'trap terminate_job TERM INT HUP' in wrapper
    assert '--resume-archive-sha256 "$expected_resume_sha256"' in wrapper
    assert "condor_hold" not in wrapper
    assert "condor_release" not in wrapper


def test_nonzero_attempt_archive_retains_resumable_triplet(
    tmp_path: Path,
) -> None:
    execution_id = "resume_failure_probe"
    worker = tmp_path / "worker_outputs"
    artifacts = worker / ".resume_work" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "checkpoint.json").write_text("checkpoint\n", encoding="utf-8")
    (artifacts / "checkpoint.estimator_call_ledger_checkpoint.abc.json").write_text(
        "ledger\n", encoding="utf-8"
    )
    (artifacts / "checkpoint.verified_singleton_resume.def.json").write_text(
        "resume\n", encoding="utf-8"
    )
    (worker / "worker_exit_status.txt").write_text("137\n", encoding="utf-8")
    job = tmp_path / f"{execution_id}.json"
    authorization = tmp_path / "authorization.json"
    activation = tmp_path / "activation.json"
    job.write_text("{}\n", encoding="utf-8")
    authorization.write_text("{}\n", encoding="utf-8")
    activation.write_text("{}\n", encoding="utf-8")
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
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["resumable_checkpoint_triplet_count"] == 1
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        receipt = json.load(archive.extractfile("worker_attempt_receipt.json"))
    assert receipt["worker_exit_status"] == 137
    assert receipt["failure_safe_checkpoint_transfer"] is True
    assert receipt["resumable_checkpoint_triplet_count"] == 1
    assert {
        "worker_outputs/.resume_work/artifacts/checkpoint.json",
        (
            "worker_outputs/.resume_work/artifacts/"
            "checkpoint.estimator_call_ledger_checkpoint.abc.json"
        ),
        (
            "worker_outputs/.resume_work/artifacts/"
            "checkpoint.verified_singleton_resume.def.json"
        ),
    } <= names
