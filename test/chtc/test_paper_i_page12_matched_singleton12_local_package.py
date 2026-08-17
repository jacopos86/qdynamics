from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_matched_singleton12_r50_20260815_v1_local"
)
RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_page12_matched_singleton12_r50_20260815.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_matched_package_is_closed_6_plus_6_without_adoption() -> None:
    contract = _load(
        "matched12_package_contract_test", PACKAGE_DIR / "package_contract.py"
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(PACKAGE_DIR / "validate_package.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert report["method_counts"] == {
        "ra_singleton_plateau": 6,
        "append_singleton": 6,
    }
    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    assert manifest["row_count"] == 12
    assert manifest["execution_order"] == [
        contract.execution_id(regime, nph, method)
        for regime, nph in contract.PAIR_EXECUTION_ORDER
        for method in contract.METHODS
    ]
    assert manifest["execution_authorized"] is False
    assert manifest["submission_authorized"] is False
    assert manifest["paper_adoption_authorized"] is False
    assert manifest["paper_evidence_adoption_authorized"] is False


def test_checkpoint_overlay_is_fresh_start_observation_only() -> None:
    contract = _load(
        "matched12_checkpoint_contract_test", PACKAGE_DIR / "package_contract.py"
    )
    worker = _load("matched12_worker_import_test", PACKAGE_DIR / "run_cell.py")
    assert worker.psutil.__name__ == "psutil"
    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    overlay = manifest["operational_checkpoint_overlay"]
    assert overlay["execution_source_policy"] == contract.EXECUTION_SOURCE_POLICY
    assert overlay["post_extraction_overlay_count"] == 1
    assert overlay["ambient_resume_overlay"] is False
    assert overlay["sealed_resume_reader_sha256"] == (
        "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
    )
    assert overlay["checkpoint_usage"] == "compact_observation_only"
    assert overlay["fresh_start_only"] is True
    assert overlay["checkpoint_resume_authorized"] is False
    assert overlay["parity_canary_scope"] == (
        "one_round_scientific_and_ledger_equivalence"
    )
    assert overlay["multi_round_compact_tail_resume_validated"] is False
    source_manifest = json.loads(
        (PACKAGE_DIR / manifest["source_archive_manifest"]["path"]).read_text()
    )
    assert source_manifest["archive_construction_no_ambient_repo_imports"] is True
    assert source_manifest["sealed_resume_reader"] == {
        "path": "pipelines/static_adapt/sr_snake/_resume.py",
        "sha256": overlay["sealed_resume_reader_sha256"],
        "size_bytes": 196_544,
        "ambient_resume_overlay": False,
    }
    for binding in manifest["jobs"]:
        job = json.loads((PACKAGE_DIR / binding["path"]).read_text())
        assert job["fresh_start_contract"]["fresh_start_only"] is True
        assert job["fresh_start_contract"]["checkpoint_resume_authorized"] is False
        assert job["checkpoint_observation"]["usage"] == (
            "compact_observation_only"
        )
        assert job["checkpoint_observation"]["resume_consumable"] is False


def test_worker_cli_is_preflight_only_and_forged_activation_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    job_path = PACKAGE_DIR / manifest["jobs"][0]["path"]
    direct = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_cell.py"),
            "--job",
            str(job_path),
            "--activation",
            str(tmp_path / "forged.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 2
    assert "--preflight" in direct.stderr or "unrecognized arguments" in direct.stderr

    worker = _load("matched12_worker_forgery_test", PACKAGE_DIR / "run_cell.py")
    with pytest.raises(worker.PackageContractError, match="pinned child seam"):
        worker.run_cell(
            job_path=job_path,
            activation_path=tmp_path / "forged.json",
            output_dir=tmp_path / "runs" / "forged",
            receipt_path=tmp_path / "receipts" / "forged.json",
            child_token="0" * 64,
        )


def test_runner_and_worker_reject_postseal_control_file_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("matched12_control_runner_test", RUNNER_PATH)
    worker = _load("matched12_control_worker_test", PACKAGE_DIR / "run_cell.py")
    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    job_path = PACKAGE_DIR / manifest["jobs"][0]["path"]

    runner_sha256_file = runner._sha256_file
    monkeypatch.setattr(
        runner,
        "_sha256_file",
        lambda path: (
            "0" * 64
            if Path(path).resolve() == (PACKAGE_DIR / "run_cell.py").resolve()
            else runner_sha256_file(Path(path))
        ),
    )
    with pytest.raises(runner.MatchedSingleton12Error, match="control file"):
        runner._package_manifest()

    worker_sha256_file = worker.sha256_file
    monkeypatch.setattr(
        worker,
        "sha256_file",
        lambda path: (
            "0" * 64
            if Path(path).resolve() == (PACKAGE_DIR / "run_cell.py").resolve()
            else worker_sha256_file(Path(path))
        ),
    )
    with pytest.raises(worker.PackageContractError, match="control file"):
        worker._load_closed_job(job_path)


def test_protocols_preserve_sealed_ra_and_matched_append_pools() -> None:
    manifest = json.loads((PACKAGE_DIR / "package_manifest.json").read_text())
    validation_binding = manifest["bundle_validation_report"]
    validation = json.loads(
        (PACKAGE_DIR / validation_binding["path"]).read_text(encoding="utf-8")
    )
    pool_rows = validation["matched_pool_receipts"]
    assert len(pool_rows) == 6
    assert all(len(row["problem_request_sha256"]) == 64 for row in pool_rows)
    assert all(row["parent_pool_count"] > 0 for row in pool_rows)
    assert all(row["executable_pool_count"] > row["parent_pool_count"] for row in pool_rows)
    protocol_rows = {
        (row["method"], row["execution_id"]): row
        for row in manifest["protocols"]
    }
    for row in pool_rows:
        regime = row["regime_id"]
        nph = row["nph"]
        ra_execution = next(
            execution
            for method, execution in protocol_rows
            if method == "ra_singleton_plateau"
            and f"__{regime}__nph{nph}__" in execution
        )
        append_execution = next(
            execution
            for method, execution in protocol_rows
            if method == "append_singleton"
            and f"__{regime}__nph{nph}__" in execution
        )
        ra = json.loads(
            (PACKAGE_DIR / protocol_rows[("ra_singleton_plateau", ra_execution)]["path"]).read_text()
        )
        append = json.loads(
            (PACKAGE_DIR / protocol_rows[("append_singleton", append_execution)]["path"]).read_text()
        )
        assert ra["sha256"] == row["ra_protocol_sha256"]
        assert append["sha256"] == row["append_protocol_sha256"]
        assert ra["problem"] == append["problem"]
        assert ra["parent_inventory"] == append["parent_inventory"]
        assert ra["executable_pool"] == append["executable_pool"]
        assert append["horizon"] == 50
        assert append["optimizer"] == "powell"
        assert append["optimizer_maxiter"] == 200
        assert append["seeds"] == {"adapt": 7, "transpiler": 7}
        assert append["selector_scope"] == "conventional_append_no_phase3_no_trust_v1"
        assert append["request"]["observation"]["checkpoint"]["keep_history_tail"] == 1
        assert append["execution_authorized"] is False
