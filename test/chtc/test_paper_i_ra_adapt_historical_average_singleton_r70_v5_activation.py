from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260802_v5_chtc"
)
ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260802_v5_chtc_activation_ordinary_v1"
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_sealed_six_row_activation_passes_read_only_validation() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "activation_id": (
            "paper_i_ra_adapt_historical_average_singleton_plateau6_"
            "r70_fresh_20260802_v5_chtc_activation_ordinary_v1"
        ),
        "activation_manifest_sha256": _load_json(
            ACTIVATION_DIR / "activation_manifest.json"
        )["sha256"],
        "batch_name": (
            "paper-i-ra-historical-average-singleton-plateau6-r70-"
            "fresh-20260802-v5"
        ),
        "direct_execution_count": 6,
        "factory": False,
        "ordinary_held": False,
        "status": "passed",
        "submission_state": "authorized_pending_remote_preflight",
    }
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))


def test_access_point_controls_do_not_require_python310_strict_zip() -> None:
    controls = (
        PACKAGE_DIR / "package_contract.py",
        PACKAGE_DIR / "validate_package.py",
        ACTIVATION_DIR / "activation_contract.py",
        ACTIVATION_DIR / "build_attempt_archive.py",
    )
    for path in controls:
        text = path.read_text(encoding="utf-8")
        assert "strict=True" not in text
        assert "strict = True" not in text


def test_queue_has_six_independent_authorities_accepted_by_worker() -> None:
    package_contract = _load_module(
        PACKAGE_DIR / "package_contract.py", "historical_average_package_contract_test"
    )
    run_cell = _load_module(PACKAGE_DIR / "run_cell.py", "historical_average_run_cell_test")
    activation_manifest = _load_json(ACTIVATION_DIR / "activation_manifest.json")
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    rows = [
        line.split("\t")
        for line in (ACTIVATION_DIR / "queue.tsv").read_text(encoding="utf-8").splitlines()
    ]
    expected_ids = list(package_contract.expected_execution_ids())
    assert len(rows) == 6
    assert [row[0] for row in rows] == expected_ids
    assert len({row[1] for row in rows}) == 6
    assert len({row[3] for row in rows}) == 6
    assert activation_manifest["direct_execution_count"] == 6
    assert len(activation_manifest["execution_authorizations"]) == 6

    for row in rows:
        execution_id, job_relative, _job_file_sha, auth_relative, *_resources = row
        job = _load_json(REPO_ROOT / job_relative)
        authorization_path = REPO_ROOT / auth_relative
        authorization = run_cell._validate_authorization(
            authorization_path,
            job=job,
            manifest=package_manifest,
        )
        assert authorization["execution_id"] == execution_id
        assert authorization["scope"] == "single_cell_chtc_execution_only"
        assert authorization["paper_evidence_adoption_authorized"] is False


def test_submit_is_lifecycle_safe_ordinary_unheld_and_attempt_unique() -> None:
    lifecycle = _load_module(
        REPO_ROOT / "chtc/validate_condor_submit_lifecycle.py",
        "historical_average_lifecycle_test",
    )
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    lifecycle.validate_submit_lifecycle(submit)
    assert "when_to_transfer_output = ON_EXIT_OR_EVICT" in submit
    assert "max_materialize" not in submit.casefold()
    assert "max_idle" not in submit.casefold()
    assert "periodic_release = False" in submit
    assert "HolsteinLifecycleMode" not in submit
    assert "hold = True" not in submit
    assert "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)" in submit
    assert "leave_in_queue = False" in submit
    assert "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz" in submit
    assert "$(Cluster).$(Process)__$(execution_id).log" in submit


def test_attempt_archive_is_deterministic_and_closed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        ACTIVATION_DIR / "build_attempt_archive.py",
        "historical_average_attempt_archive_test",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = "historical_average_v4_r70_fresh__weak_weak__nph3__ra_singleton_plateau"
    worker_root = Path("worker_outputs")
    artifacts = worker_root / "artifacts"
    artifacts.mkdir(parents=True)
    (worker_root / "attempt_identity.tsv").write_text(
        f"{execution_id}\t123\t4\t1\n", encoding="utf-8"
    )
    (worker_root / "worker_exit_status.txt").write_text("0\n", encoding="utf-8")
    (worker_root / "worker_receipt.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "checkpoint.json",
        "estimator_ledger.json",
        "result.json",
        "execution_manifest.json",
    ):
        (artifacts / name).write_text("{}\n", encoding="utf-8")
    job = Path(f"{execution_id}.json")
    authorization = Path("execution_authorization.json")
    activation_manifest = Path("activation_manifest.json")
    job.write_text("{}\n", encoding="utf-8")
    authorization.write_text("{}\n", encoding="utf-8")
    activation_manifest.write_text("{}\n", encoding="utf-8")
    Path("transfer").mkdir()
    output = Path("transfer/attempt.tar.gz")
    args = argparse.Namespace(
        worker_root=worker_root,
        job=job,
        authorization=authorization,
        activation_manifest=activation_manifest,
        output_archive=output,
        execution_id=execution_id,
        cluster_id=123,
        proc_id=4,
        attempt_ordinal=1,
        worker_exit_status=0,
        source_archive_sha256="1" * 64,
        image_sha256="2" * 64,
    )
    first = builder.build_archive(args)
    first_bytes = output.read_bytes()
    output.unlink()
    second = builder.build_archive(args)
    assert output.read_bytes() == first_bytes
    assert second["output_archive_sha256"] == first["output_archive_sha256"]
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        receipt = json.load(archive.extractfile("worker_attempt_receipt.json"))
    assert names == [
        "worker_outputs/artifacts/checkpoint.json",
        "worker_outputs/artifacts/estimator_ledger.json",
        "worker_outputs/artifacts/execution_manifest.json",
        "worker_outputs/artifacts/result.json",
        "worker_outputs/attempt_identity.tsv",
        "worker_outputs/worker_exit_status.txt",
        "worker_outputs/worker_receipt.json",
        "authority/job.json",
        "authority/execution_authorization.json",
        "authority/activation_manifest.json",
        "worker_attempt_receipt.json",
    ]
    assert receipt["schema"] == builder.ATTEMPT_SCHEMA
    assert receipt["execution_id"] == execution_id
