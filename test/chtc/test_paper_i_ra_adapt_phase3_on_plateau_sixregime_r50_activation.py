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

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc"
)
ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc_activation_ordinary_v1"
)
EXPECTED_IDS = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_weak_u8__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__weak_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_strong_u8__nph7__ra_singleton_plateau",
)
EXPECTED_RESOURCES = (
    ("4", "24576", "40960", "259200"),
    ("4", "24576", "40960", "259200"),
    ("4", "24576", "40960", "259200"),
    ("4", "32768", "61440", "259200"),
    ("4", "32768", "61440", "259200"),
    ("4", "32768", "61440", "259200"),
)
ROUTE_CONTRACT_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
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


def _closed_ledger() -> dict[str, Any]:
    components = {
        "N_H_outer": 50,
        "N_H_refit": 20,
        "N_grad": 30,
        "N_metric": 40,
    }
    return {
        "schema": "paper_i_estimator_call_ledger_sidecar_v2",
        "accounting": {
            "complete": True,
            "exact_blockers": [],
            "components": components,
            "S_alg": sum(components.values()),
        },
        "ledger": {},
        "adapt_success": True,
        "adapt_error": None,
    }


def test_activation_passes_read_only_validation_and_binds_final_package() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    manifest = _load_json(ACTIVATION_DIR / "activation_manifest.json")
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")

    assert result == {
        "activation_id": (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
            "r50_20260803_v1_chtc_activation_ordinary_v1"
        ),
        "activation_manifest_sha256": manifest["sha256"],
        "batch_name": (
            "paper-i-ra-phase3-on-plateau-sixregime-r50-20260803-v1"
        ),
        "direct_execution_count": 6,
        "factory": False,
        "ordinary_held": False,
        "status": "passed",
        "submission_state": "authorized_pending_remote_preflight",
    }
    assert manifest["sealed_package"]["manifest"]["canonical_sha256"] == (
        package_manifest["sha256"]
    )
    assert manifest["sealed_package"]["manifest"]["sha256"] == (
        "46a488ad7a68f1b9865515c01898a93302af8d4f1abb8c988b4f14b3f165c73e"
    )
    assert manifest["sealed_package"]["source_archive"]["sha256"] == (
        "e388f39093664ad7a342907f7e604e0a100673a0abb60d79923947b199af006b"
    )
    assert manifest["remote_image"] == {
        "path": "chtc/phase3_optuna/image.sif",
        "sha256": IMAGE_SHA256,
        "byte_verification_required_before_submit": True,
        "byte_verification_passed": False,
    }
    assert manifest["known_transfer_limitations"] == [
        (
            "partial_checkpoint_and_ledger_are_not_preserved_before_"
            "success_publication_v1"
        ),
        (
            "retry_attempts_reuse_the_cluster_proc_fetched_archive_"
            "destination_v1"
        ),
    ]
    assert not list(ACTIVATION_DIR.rglob("__pycache__"))


def test_queue_has_six_independent_authorities_and_exact_resource_rows() -> None:
    run_cell = _load_module(
        PACKAGE_DIR / "run_cell.py",
        "phase3_on_plateau_activation_run_cell_test",
    )
    activation_manifest = _load_json(ACTIVATION_DIR / "activation_manifest.json")
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    rows = [
        line.split("\t")
        for line in (ACTIVATION_DIR / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(rows) == 6
    assert tuple(row[0] for row in rows) == EXPECTED_IDS
    assert tuple(tuple(row[5:]) for row in rows) == EXPECTED_RESOURCES
    assert len({row[1] for row in rows}) == 6
    assert len({row[3] for row in rows}) == 6
    assert activation_manifest["direct_execution_count"] == 6
    assert len(activation_manifest["execution_authorizations"]) == 6

    for row in rows:
        execution_id, job_relative, _job_file_sha, auth_relative, *_ = row
        job = _load_json(REPO_ROOT / job_relative)
        authorization = run_cell._validate_authorization(
            REPO_ROOT / auth_relative,
            job=job,
            manifest=package_manifest,
        )
        assert job["route_contract_sha256"] == ROUTE_CONTRACT_SHA256
        assert job["execution_mode"] == "fresh_0_to_50"
        assert job["target_horizon"] == 50
        assert authorization["execution_id"] == execution_id
        assert authorization["scope"] == "single_cell_chtc_execution_only"
        assert authorization["paper_evidence_adoption_authorized"] is False
        assert authorization["remote_image_sha256"] == IMAGE_SHA256


def test_submit_is_lifecycle_safe_ordinary_unheld_with_runtime_paths() -> None:
    lifecycle = _load_module(
        REPO_ROOT / "chtc/validate_condor_submit_lifecycle.py",
        "phase3_on_plateau_activation_lifecycle_test",
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
    assert "chtc/phase3_optuna/image.sif" in submit
    assert IMAGE_SHA256 in submit
    assert "$(execution_id)__$(ClusterId)__$(ProcId).tar.gz" in submit
    assert "$(Cluster).$(Process)__$(execution_id).log" in submit
    assert (
        "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
        "r50_20260803_v1_chtc_runtime/fetched/"
    ) in submit


def test_success_attempt_archive_is_deterministic_and_evidence_closed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    builder = _load_module(
        ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_on_plateau_attempt_archive_test",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = EXPECTED_IDS[0]
    worker_root = Path("worker_outputs")
    artifacts = worker_root / "artifacts"
    artifacts.mkdir(parents=True)
    (worker_root / "attempt_identity.tsv").write_text(
        f"{execution_id}\t123\t4\t1\n", encoding="utf-8"
    )
    (worker_root / "worker_exit_status.txt").write_text(
        "0\n", encoding="utf-8"
    )
    (worker_root / "worker_receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    for name in (
        "checkpoint.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    ):
        (artifacts / name).write_text("{}\n", encoding="utf-8")
    (artifacts / "estimator_ledger.json").write_text(
        json.dumps(_closed_ledger()) + "\n",
        encoding="utf-8",
    )
    job = Path(f"{execution_id}.json")
    authorization = Path("execution_authorization.json")
    activation_manifest = Path("activation_manifest.json")
    for path in (job, authorization, activation_manifest):
        path.write_text("{}\n", encoding="utf-8")
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
        "worker_outputs/artifacts/paper_i_summary.json",
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


@pytest.mark.parametrize("failure_kind", ("missing_summary", "open_ledger"))
def test_success_attempt_archive_rejects_incomplete_evidence(
    tmp_path: Path, monkeypatch: Any, failure_kind: str
) -> None:
    builder = _load_module(
        ACTIVATION_DIR / "build_attempt_archive.py",
        f"phase3_on_plateau_attempt_rejection_{failure_kind}",
    )
    monkeypatch.chdir(tmp_path)
    execution_id = EXPECTED_IDS[0]
    worker_root = Path("worker_outputs")
    artifacts = worker_root / "artifacts"
    artifacts.mkdir(parents=True)
    for name in (
        "checkpoint.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    ):
        (artifacts / name).write_text("{}\n", encoding="utf-8")
    ledger = _closed_ledger()
    if failure_kind == "missing_summary":
        (artifacts / "paper_i_summary.json").unlink()
    else:
        ledger["accounting"]["complete"] = False
    (artifacts / "estimator_ledger.json").write_text(
        json.dumps(ledger) + "\n",
        encoding="utf-8",
    )
    job = Path(f"{execution_id}.json")
    authorization = Path("execution_authorization.json")
    activation_manifest = Path("activation_manifest.json")
    for path in (job, authorization, activation_manifest):
        path.write_text("{}\n", encoding="utf-8")
    Path("transfer").mkdir()
    args = argparse.Namespace(
        worker_root=worker_root,
        job=job,
        authorization=authorization,
        activation_manifest=activation_manifest,
        output_archive=Path("transfer/attempt.tar.gz"),
        execution_id=execution_id,
        cluster_id=123,
        proc_id=4,
        attempt_ordinal=1,
        worker_exit_status=0,
        source_archive_sha256="1" * 64,
        image_sha256="2" * 64,
    )

    with pytest.raises(builder.AttemptArchiveError):
        builder.build_archive(args)
