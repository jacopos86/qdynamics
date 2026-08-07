from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
PARENT = REPO / "chtc/phase3_optuna/input/paper_i_hh_geo_projected_singleton_all_six_r50_20260719_v5_chtc"
SOURCE_SHA256 = "8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd"
JOB_IDS = (
    "geo_projected_singleton__intermediate_strong__r50",
    "geo_projected_singleton__strong_strong__r50",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_two_row_parent_science_contract() -> None:
    manifest = _load(BUNDLE / "bundle_manifest.json")
    assert manifest["status"] == "prepared_not_submitted"
    assert manifest["job_count"] == 2
    assert manifest["scientific_settings_changed"] == []
    assert manifest["parent"]["cluster_id"] == 8887546
    assert manifest["parent"]["held_procs_replaced"] == [4, 5]
    lines = (BUNDLE / "queue.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [line.split("\t", 1)[0] for line in lines] == list(JOB_IDS)
    for job_id in JOB_IDS:
        for directory in ("jobs", "normalized_manifests"):
            local = BUNDLE / directory / f"{job_id}.json"
            parent = PARENT / directory / f"{job_id}.json"
            assert local.read_bytes() == parent.read_bytes()
            assert _load(local)["bundle_id"] == PARENT.name
    assert (BUNDLE / "run_job.py").read_bytes() == (PARENT / "run_job.py").read_bytes()
    assert _sha256(BUNDLE / "source_locked.tar.gz") == SOURCE_SHA256
    assert (BUNDLE / "source_locked.tar.gz").read_bytes() == (
        PARENT / "source_locked.tar.gz"
    ).read_bytes()


def test_submit_is_narrow_and_bundle_local() -> None:
    submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
    assert "packaging-repair-20260719-v6" in submit
    assert f"chtc/phase3_optuna/input/{BUNDLE.name}/queue.tsv" in submit
    assert f"raw_outputs/{BUNDLE.name}/$(job_id)_transfer.tar.gz" in submit
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in submit
    assert "requirements = TARGET.HasSIF" in submit


def test_wrapper_always_packages_failed_execution(tmp_path: Path) -> None:
    source_root = tmp_path / "source_root"
    source_root.mkdir()
    (source_root / "sentinel.txt").write_text("source", encoding="utf-8")
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        archive.add(source_root / "sentinel.txt", arcname="sentinel.txt")
    image = tmp_path / "image.sif"
    image.write_bytes(b"fake-image")
    manifest = tmp_path / "job.json"
    manifest.write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_apptainer = fake_bin / "apptainer"
    fake_apptainer.write_text("#!/usr/bin/env bash\nexit 23\n", encoding="utf-8")
    fake_apptainer.chmod(0o755)
    job_id = JOB_IDS[0]
    result = subprocess.run(
        [
            str(BUNDLE / "execute_source_locked_job.sh"),
            str(manifest),
            str(source_archive),
            _sha256(source_archive),
            str(image),
            _sha256(image),
            job_id,
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 23
    transfer = tmp_path / "raw_outputs" / BUNDLE.name / f"{job_id}_transfer.tar.gz"
    assert transfer.is_file()
    with tarfile.open(transfer, "r:gz") as archive:
        names = {member.name.rstrip("/") for member in archive.getmembers()}
    assert job_id in names


def test_submission_artifact_hash_inventory_matches() -> None:
    payload = _load(BUNDLE / "submission_artifact_hashes.json")
    for relative, expected in payload["files"].items():
        assert _sha256(BUNDLE / relative) == expected

