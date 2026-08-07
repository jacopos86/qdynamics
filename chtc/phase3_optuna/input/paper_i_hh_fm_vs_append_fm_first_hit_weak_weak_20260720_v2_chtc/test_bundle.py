from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
BUNDLE_ID = BUNDLE.name
JOB_ID = "weak_weak_fm_vs_append_fm_first_hit"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_is_one_sequential_pair_and_not_submitted() -> None:
    manifest = _load(BUNDLE / "bundle_manifest.json")
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["submission_performed"] is False
    assert manifest["job_count"] == 1
    assert manifest["condor_proc_count"] == 1
    assert manifest["sequential_route_count_within_proc"] == 2
    assert manifest["reporting_scope"] == {
        "query_coordinate": "winning_lineage_S_alg_only",
        "discarded_branch_work_reported": False,
    }
    preflight = _load(BUNDLE / "preflight.json")
    assert preflight["status"] == "pass"
    assert preflight["checks"]["scientific_execution_performed"] is False
    assert preflight["checks"]["chtc_submission_performed"] is False


def test_job_and_queue_lock_weak_weak_first_hit_contract() -> None:
    job = _load(BUNDLE / "jobs" / f"{JOB_ID}.json")
    normalized = _load(BUNDLE / "normalized_manifests" / f"{JOB_ID}.json")
    assert job == normalized
    assert job["regime"] == "weak-weak"
    assert job["physics"]["n_ph_work"] == 3
    assert job["physics"]["same_cutoff"] is True
    comparison = job["comparison"]
    assert comparison["initial_ansatz"] == "empty_hf_reference_v1"
    assert comparison["automatic_hh_seed_disabled"] is True
    assert comparison["routes"] == ["fm_snake", "projected_singleton_append_fm"]
    assert comparison["target_abs_delta_e"] == 2.0e-4
    assert comparison["max_controller_rounds"] == 30
    assert comparison["optimizer_maxiter"] == 200
    assert comparison["line_search_max_steps"] == 15
    assert comparison["qbroyd_qbang_enabled"] is False
    assert comparison["reported_query_coordinate"] == "winning_lineage_S_alg_only"
    assert comparison["discarded_branch_work_reported"] is False
    with (BUNDLE / "queue.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(
            csv.DictReader(
                handle,
                fieldnames=("job_id", "job_manifest", "normalized_manifest"),
                delimiter="\t",
            )
        )
    assert len(rows) == 1
    assert rows[0]["job_id"] == JOB_ID
    assert (REPO / rows[0]["job_manifest"]).is_file()
    assert (REPO / rows[0]["normalized_manifest"]).is_file()


def test_submit_is_one_nonstreaming_4cpu_24gb_40gb_proc() -> None:
    submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
    assert "request_cpus = 4" in submit
    assert "request_memory = 24576MB" in submit
    assert "request_disk = 40960MB" in submit
    assert "stream_output = False" in submit
    assert "stream_error = False" in submit
    assert "requirements = TARGET.HasSIF" in submit
    assert "queue job_id, job_manifest, normalized_manifest from" in submit
    assert "transfer_output_files = raw_outputs/" in submit
    assert "_transfer.tar.gz" in submit


def test_failure_transfer_preserves_only_narrow_recovery_checkpoints() -> None:
    wrapper = (BUNDLE / "execute_source_locked_job.sh").read_text(encoding="utf-8")
    assert 'if [[ "$status" -ne 0 ]]' in wrapper
    assert '"weak-weak/fm_snake/current.json"' in wrapper
    assert '"weak-weak/projected_singleton_append_fm/partial_result.json"' in wrapper
    assert (
        '"weak-weak/projected_singleton_append_fm/adapt_iteration_progress.jsonl"'
        in wrapper
    )
    assert 'tar -czf "${TRANSFER_ARCHIVE}.tmp" "$TRANSFER_ROOT"' in wrapper


def test_source_archive_inventory_and_cli_import() -> None:
    manifest = _load(BUNDLE / "source_archive_manifest.json")
    archive = BUNDLE / "source_locked.tar.gz"
    assert manifest["archive_sha256"] == _sha256(archive)
    assert manifest["file_count"] == len(manifest["files"])
    module_path = "pipelines/exact_bench/paper_i_hh_fm_vs_append_fm_first_hit.py"
    assert module_path in manifest["files"]
    with tarfile.open(archive, "r:gz") as tar:
        members = {member.name: member for member in tar.getmembers() if member.isfile()}
        assert set(members) == set(manifest["files"])
        for relative, record in manifest["files"].items():
            stream = tar.extractfile(members[relative])
            assert stream is not None
            assert hashlib.sha256(stream.read()).hexdigest() == record["sha256"]
        with tempfile.TemporaryDirectory(prefix="fm-append-bundle-test-") as tmp_name:
            root = Path(tmp_name)
            tar.extractall(root, filter="data")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pipelines.exact_bench.paper_i_hh_fm_vs_append_fm_first_hit",
                    "--help",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert proc.returncode == 0, proc.stderr
            assert "run-pair" in proc.stdout


def test_submission_hashes_cover_all_static_artifacts() -> None:
    hashes = _load(BUNDLE / "submission_artifact_hashes.json")["files"]
    required = {
        "README.md",
        "build_bundle.py",
        "bundle_manifest.json",
        "execute_source_locked_job.sh",
        f"jobs/{JOB_ID}.json",
        f"normalized_manifests/{JOB_ID}.json",
        "preflight.json",
        "queue.tsv",
        "run_job.py",
        "source_archive_manifest.json",
        "source_locked.tar.gz",
        "source_revision_manifest.json",
        "submit.sub",
        "test_bundle.py",
    }
    assert required <= set(hashes)
    for relative, expected in hashes.items():
        assert _sha256(BUNDLE / relative) == expected
