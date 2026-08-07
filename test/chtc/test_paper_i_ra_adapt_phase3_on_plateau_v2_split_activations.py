from __future__ import annotations

import argparse
import hashlib
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
    "r50_20260803_v2_chtc"
)
ORDINARY_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_ordinary_v1"
)
CANARY_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_canary_weak_strong_v1"
)
REMAINING_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc_activation_remaining5_v1"
)
CANARY_ID = (
    "phase3_on_plateau_r50__weak_strong__nph7__ra_singleton_plateau"
)
REMAINING_IDS = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_weak__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_weak_u8__nph3__ra_singleton_plateau",
    "phase3_on_plateau_r50__intermediate_strong__nph7__ra_singleton_plateau",
    "phase3_on_plateau_r50__strong_strong_u8__nph7__ra_singleton_plateau",
)
EXPECTED_RESOURCES = {
    REMAINING_IDS[0]: ("4", "24576", "40960", "259200"),
    REMAINING_IDS[1]: ("4", "24576", "40960", "259200"),
    REMAINING_IDS[2]: ("4", "24576", "40960", "259200"),
    CANARY_ID: ("4", "32768", "61440", "259200"),
    REMAINING_IDS[3]: ("4", "32768", "61440", "259200"),
    REMAINING_IDS[4]: ("4", "32768", "61440", "259200"),
}
CONTROL_FILES = {
    "activation_contract.py",
    "materialize_activation.py",
    "validate_activation.py",
    "execute_authorized_job.sh",
    "build_attempt_archive.py",
    "submit.sub.in",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digested(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


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


def _write_success_worker(
    *,
    execution_id: str,
    job: dict[str, Any],
    authorization: dict[str, Any],
    package_manifest: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    worker_root = Path("worker_outputs")
    artifacts = worker_root / "artifacts"
    artifacts.mkdir(parents=True)

    estimator_sidecar = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
        "checkpoint": {"depth": 50, "current_round_finalized": True},
        "ledger": {},
        "no_credentials_serialized": True,
    }
    estimator_bytes = (
        json.dumps(estimator_sidecar, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    estimator_sha = hashlib.sha256(estimator_bytes).hexdigest()
    estimator_name = (
        "checkpoint.estimator_call_ledger_checkpoint."
        f"{estimator_sha[:16]}.json"
    )
    (artifacts / estimator_name).write_bytes(estimator_bytes)

    resume_sidecar = {
        "schema": "static_adapt_signed_active_prefix_resume_sidecar_v2",
        "source": {"checkpoint_depth": 50},
        "no_credentials_serialized": True,
    }
    resume_bytes = (
        json.dumps(resume_sidecar, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    resume_sha = hashlib.sha256(resume_bytes).hexdigest()
    resume_name = (
        "checkpoint.verified_singleton_resume."
        f"{resume_sha[:16]}.json"
    )
    (artifacts / resume_name).write_bytes(resume_bytes)

    _write_json(
        artifacts / "checkpoint.json",
        {
            "schema_version": "static_adapt_current_checkpoint_v1",
            "checkpoint": {"depth": 50},
            "adapt_vqe": {
                "estimator_call_ledger_checkpoint": {
                    "path": estimator_name,
                    "sha256": estimator_sha,
                },
                "verified_singleton_resume_sidecar": {
                    "path": resume_name,
                    "sha256": resume_sha,
                },
            },
        },
    )
    _write_json(artifacts / "estimator_ledger.json", _closed_ledger())
    _write_json(
        artifacts / "paper_i_summary.json",
        {"schema": "paper_i_run_summary_v1"},
    )
    _write_json(
        artifacts / "result.json",
        {"schema": "ra_adapt_result_v1"},
    )

    preliminary = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(artifacts.iterdir())
    }
    execution_manifest = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_singleton_phase3_on_plateau_"
                "sixregime_r50_execution_manifest_v2"
            ),
            "status": "passed",
            "package_id": package_manifest["package_id"],
            "campaign_id": package_manifest["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "target_horizon": 50,
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "worker_owned_live_progress": True,
            "same_filesystem_atomic_success_publication": True,
            "output_payloads": preliminary,
        }
    )
    _write_json(artifacts / "execution_manifest.json", execution_manifest)

    receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_singleton_phase3_on_plateau_"
                "sixregime_r50_worker_receipt_v2"
            ),
            "status": "passed",
            "package_id": package_manifest["package_id"],
            "campaign_id": package_manifest["campaign_id"],
            "execution_id": execution_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": [
                _binding(path) for path in sorted(artifacts.iterdir())
            ],
        }
    )
    _write_json(worker_root / "worker_receipt.json", receipt)
    (worker_root / "attempt_identity.tsv").write_text(
        f"{execution_id}\t123\t4\t1\n",
        encoding="utf-8",
    )
    (worker_root / "worker_exit_status.txt").write_text(
        "0\n",
        encoding="utf-8",
    )
    return worker_root, receipt


def _attempt_args(
    *,
    execution_id: str,
    job_path: Path,
    authorization_path: Path,
    worker_exit_status: int,
) -> argparse.Namespace:
    Path("transfer").mkdir(exist_ok=True)
    authorization = _load_json(authorization_path)
    activation_manifest = (
        REPAIR_ROOT
        / authorization["activation_id"]
        / "activation_manifest.json"
    )
    return argparse.Namespace(
        worker_root=Path("worker_outputs"),
        job=job_path,
        authorization=authorization_path,
        activation_manifest=activation_manifest,
        output_archive=Path("transfer/attempt.tar.gz"),
        execution_id=execution_id,
        cluster_id=123,
        proc_id=4,
        attempt_ordinal=1,
        worker_exit_status=worker_exit_status,
        source_archive_sha256=authorization["source_archive_sha256"],
        image_sha256=authorization["remote_image_sha256"],
    )


def _queue_rows(activation_dir: Path) -> list[list[str]]:
    return [
        line.split("\t")
        for line in (activation_dir / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _validate(activation_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-B", str(activation_dir / "validate_activation.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    prior_package_contract = sys.modules.pop("package_contract", None)
    sys.path.insert(0, path.parent.as_posix())
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
        sys.path.remove(path.parent.as_posix())
        sys.modules.pop("package_contract", None)
        if prior_package_contract is not None:
            sys.modules["package_contract"] = prior_package_contract
    return module


def test_split_activations_validate_and_partition_the_sealed_package() -> None:
    canary_receipt = _validate(CANARY_ACTIVATION_DIR)
    remaining_receipt = _validate(REMAINING_ACTIVATION_DIR)
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    canary_manifest = _load_json(
        CANARY_ACTIVATION_DIR / "activation_manifest.json"
    )
    remaining_manifest = _load_json(
        REMAINING_ACTIVATION_DIR / "activation_manifest.json"
    )
    canary_rows = _queue_rows(CANARY_ACTIVATION_DIR)
    remaining_rows = _queue_rows(REMAINING_ACTIVATION_DIR)

    assert canary_receipt["status"] == "passed"
    assert remaining_receipt["status"] == "passed"
    assert canary_receipt["direct_execution_count"] == 1
    assert remaining_receipt["direct_execution_count"] == 5
    assert tuple(row[0] for row in canary_rows) == (CANARY_ID,)
    assert tuple(row[0] for row in remaining_rows) == REMAINING_IDS
    assert set(row[0] for row in canary_rows).isdisjoint(
        row[0] for row in remaining_rows
    )
    assert set(row[0] for row in (*canary_rows, *remaining_rows)) == set(
        package_manifest["execution_ids"]
    )
    package_order_rows = (
        *remaining_rows[:3],
        *canary_rows,
        *remaining_rows[3:],
    )
    assert tuple(row[0] for row in package_order_rows) == tuple(
        package_manifest["execution_ids"]
    )
    for row in (*canary_rows, *remaining_rows):
        assert tuple(row[5:]) == EXPECTED_RESOURCES[row[0]]

    assert canary_manifest["package_execution_count"] == 6
    assert remaining_manifest["package_execution_count"] == 6
    assert canary_manifest["activation_execution_ids"] == [CANARY_ID]
    assert remaining_manifest["activation_execution_ids"] == list(
        REMAINING_IDS
    )
    assert canary_manifest["row_selection"] == "weak_strong_canary_only_v1"
    assert remaining_manifest["row_selection"] == (
        "remaining_five_excluding_weak_strong_canary_v1"
    )


def test_split_activations_share_package_and_repaired_worker_bytes() -> None:
    canary_manifest = _load_json(
        CANARY_ACTIVATION_DIR / "activation_manifest.json"
    )
    remaining_manifest = _load_json(
        REMAINING_ACTIVATION_DIR / "activation_manifest.json"
    )
    ordinary_manifest = _load_json(
        ORDINARY_ACTIVATION_DIR / "activation_manifest.json"
    )
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")

    assert canary_manifest["sealed_package"] == remaining_manifest[
        "sealed_package"
    ]
    sealed = canary_manifest["sealed_package"]
    assert sealed["manifest"]["canonical_sha256"] == package_manifest["sha256"]
    for name in ("execute_authorized_job.sh", "build_attempt_archive.py"):
        expected = (ORDINARY_ACTIVATION_DIR / name).read_bytes()
        assert (CANARY_ACTIVATION_DIR / name).read_bytes() == expected
        assert (REMAINING_ACTIVATION_DIR / name).read_bytes() == expected

    assert canary_manifest["known_transfer_limitations"] == ordinary_manifest[
        "known_transfer_limitations"
    ]
    assert remaining_manifest["known_transfer_limitations"] == (
        canary_manifest["known_transfer_limitations"]
    )


def test_split_activations_have_distinct_unheld_runtime_control_planes() -> None:
    lifecycle = _load_module(
        REPO_ROOT / "chtc/validate_condor_submit_lifecycle.py",
        "phase3_on_plateau_v2_split_lifecycle",
    )
    canary_manifest = _load_json(
        CANARY_ACTIVATION_DIR / "activation_manifest.json"
    )
    remaining_manifest = _load_json(
        REMAINING_ACTIVATION_DIR / "activation_manifest.json"
    )
    canary_submit = (CANARY_ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )
    remaining_submit = (REMAINING_ACTIVATION_DIR / "submit.sub").read_text(
        encoding="utf-8"
    )

    lifecycle.validate_submit_lifecycle(canary_submit)
    lifecycle.validate_submit_lifecycle(remaining_submit)
    assert canary_manifest["activation_id"] != remaining_manifest["activation_id"]
    assert canary_manifest["batch_name"] != remaining_manifest["batch_name"]
    assert canary_submit != remaining_submit
    assert (
        "r50_20260803_v2_chtc_runtime_canary_weak_strong_v1/" in canary_submit
    )
    assert "r50_20260803_v2_chtc_runtime_remaining5_v1/" in remaining_submit
    for manifest, submit in (
        (canary_manifest, canary_submit),
        (remaining_manifest, remaining_submit),
    ):
        assert manifest["operational_mode"] == (
            "ordinary_unheld_worker_durable_v2"
        )
        assert manifest["remote_stage"] is False
        assert manifest["condor_submit"] is False
        assert manifest["submitted"] is False
        assert f'+JobBatchName = "{manifest["batch_name"]}"' in submit
        assert "max_materialize" not in submit.casefold()
        assert "max_idle" not in submit.casefold()
        assert "hold = True" not in submit
        assert "periodic_release = False" in submit
        assert "leave_in_queue = False" in submit


def test_split_authorizations_are_accepted_by_final_v2_worker() -> None:
    run_cell = _load_module(
        PACKAGE_DIR / "run_cell.py",
        "phase3_on_plateau_v2_split_run_cell",
    )
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")

    for activation_dir in (CANARY_ACTIVATION_DIR, REMAINING_ACTIVATION_DIR):
        activation_manifest = _load_json(
            activation_dir / "activation_manifest.json"
        )
        for row in _queue_rows(activation_dir):
            execution_id, job_relative, _job_sha, auth_relative, *_ = row
            job = _load_json(REPO_ROOT / job_relative)
            authorization = run_cell._validate_authorization(
                REPO_ROOT / auth_relative,
                job=job,
                manifest=package_manifest,
            )
            assert authorization["execution_id"] == execution_id
            assert authorization["activation_id"] == activation_manifest[
                "activation_id"
            ]
            assert authorization["paper_evidence_adoption_authorized"] is False


def test_split_activation_file_closures_are_exact() -> None:
    for activation_dir, execution_ids in (
        (CANARY_ACTIVATION_DIR, (CANARY_ID,)),
        (REMAINING_ACTIVATION_DIR, REMAINING_IDS),
    ):
        actual = {
            path.relative_to(activation_dir).as_posix()
            for path in activation_dir.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        expected = CONTROL_FILES | {
            "activation_manifest.json",
            "queue.tsv",
            "submit.sub",
            *(f"authorizations/{execution_id}.json" for execution_id in execution_ids),
        }
        assert actual == expected
        assert not list(activation_dir.rglob("__pycache__"))


def test_split_success_archive_accepts_fully_authenticated_worker_receipt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    builder = _load_module(
        CANARY_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_on_plateau_v2_split_receipt_success",
    )
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    canary_row = _queue_rows(CANARY_ACTIVATION_DIR)[0]
    job_path = REPO_ROOT / canary_row[1]
    authorization_path = REPO_ROOT / canary_row[3]
    job = _load_json(job_path)
    authorization = _load_json(authorization_path)
    monkeypatch.chdir(tmp_path)
    _write_success_worker(
        execution_id=CANARY_ID,
        job=job,
        authorization=authorization,
        package_manifest=package_manifest,
    )
    args = _attempt_args(
        execution_id=CANARY_ID,
        job_path=job_path,
        authorization_path=authorization_path,
        worker_exit_status=0,
    )

    result = builder.build_archive(args)
    assert result["status"] == "passed"
    with tarfile.open(args.output_archive, "r:gz") as archive:
        attempt_receipt = json.load(
            archive.extractfile("worker_attempt_receipt.json")
        )
    assert attempt_receipt["science_evidence_state"] == (
        "success_payload_closed_v2"
    )


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "missing",
        "self_digest",
        "execution_manifest_binding",
        "artifact_binding",
    ),
)
def test_split_success_archive_rejects_missing_or_tampered_worker_receipt(
    tmp_path: Path,
    monkeypatch: Any,
    tamper_kind: str,
) -> None:
    builder = _load_module(
        CANARY_ACTIVATION_DIR / "build_attempt_archive.py",
        f"phase3_on_plateau_v2_split_receipt_{tamper_kind}",
    )
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    canary_row = _queue_rows(CANARY_ACTIVATION_DIR)[0]
    job_path = REPO_ROOT / canary_row[1]
    authorization_path = REPO_ROOT / canary_row[3]
    job = _load_json(job_path)
    authorization = _load_json(authorization_path)
    monkeypatch.chdir(tmp_path)
    worker_root, receipt = _write_success_worker(
        execution_id=CANARY_ID,
        job=job,
        authorization=authorization,
        package_manifest=package_manifest,
    )
    receipt_path = worker_root / "worker_receipt.json"
    if tamper_kind == "missing":
        receipt_path.unlink()
    elif tamper_kind == "self_digest":
        receipt["sha256"] = "0" * 64
        _write_json(receipt_path, receipt)
    elif tamper_kind == "execution_manifest_binding":
        receipt["execution_manifest_sha256"] = "0" * 64
        _write_json(receipt_path, _digested(receipt))
    else:
        receipt["artifacts"][0]["sha256"] = "0" * 64
        _write_json(receipt_path, _digested(receipt))

    args = _attempt_args(
        execution_id=CANARY_ID,
        job_path=job_path,
        authorization_path=authorization_path,
        worker_exit_status=0,
    )
    with pytest.raises(builder.AttemptArchiveError):
        builder.build_archive(args)
    assert not args.output_archive.exists()


@pytest.mark.parametrize(
    "tamper_kind",
    (
        "self_digest",
        "package",
        "campaign",
        "execution",
        "job",
        "authorization",
        "protocol",
        "horizon",
        "artifact_payloads",
    ),
)
def test_split_success_archive_authenticates_execution_manifest_authority(
    tmp_path: Path,
    monkeypatch: Any,
    tamper_kind: str,
) -> None:
    builder = _load_module(
        REMAINING_ACTIVATION_DIR / "build_attempt_archive.py",
        f"phase3_on_plateau_v2_split_manifest_{tamper_kind}",
    )
    package_manifest = _load_json(PACKAGE_DIR / "package_manifest.json")
    canary_row = _queue_rows(CANARY_ACTIVATION_DIR)[0]
    job_path = REPO_ROOT / canary_row[1]
    authorization_path = REPO_ROOT / canary_row[3]
    job = _load_json(job_path)
    authorization = _load_json(authorization_path)
    monkeypatch.chdir(tmp_path)
    worker_root, receipt = _write_success_worker(
        execution_id=CANARY_ID,
        job=job,
        authorization=authorization,
        package_manifest=package_manifest,
    )
    execution_manifest_path = worker_root / "artifacts/execution_manifest.json"
    execution_manifest = _load_json(execution_manifest_path)
    if tamper_kind == "self_digest":
        execution_manifest["sha256"] = "0" * 64
    elif tamper_kind == "package":
        execution_manifest["package_id"] = "wrong_package"
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "campaign":
        execution_manifest["campaign_id"] = "wrong_campaign"
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "execution":
        execution_manifest["execution_id"] = "wrong_execution"
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "job":
        execution_manifest["job_spec_sha256"] = "0" * 64
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "authorization":
        execution_manifest["authorization_sha256"] = "0" * 64
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "protocol":
        execution_manifest["protocol_sha256"] = "0" * 64
        execution_manifest = _digested(execution_manifest)
    elif tamper_kind == "horizon":
        execution_manifest["target_horizon"] = 49
        execution_manifest = _digested(execution_manifest)
    else:
        execution_manifest["output_payloads"]["result.json"][
            "sha256"
        ] = "0" * 64
        execution_manifest = _digested(execution_manifest)
    _write_json(execution_manifest_path, execution_manifest)

    receipt["execution_manifest_sha256"] = execution_manifest["sha256"]
    receipt["artifacts"] = [
        _binding(path)
        for path in sorted((worker_root / "artifacts").iterdir())
    ]
    _write_json(worker_root / "worker_receipt.json", _digested(receipt))
    args = _attempt_args(
        execution_id=CANARY_ID,
        job_path=job_path,
        authorization_path=authorization_path,
        worker_exit_status=0,
    )

    with pytest.raises(builder.AttemptArchiveError):
        builder.build_archive(args)
    assert not args.output_archive.exists()


def test_split_failure_archive_preserves_unauthenticated_progress(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    builder = _load_module(
        REMAINING_ACTIVATION_DIR / "build_attempt_archive.py",
        "phase3_on_plateau_v2_split_failure_progress",
    )
    remaining_row = _queue_rows(REMAINING_ACTIVATION_DIR)[0]
    job_path = REPO_ROOT / remaining_row[1]
    authorization_path = REPO_ROOT / remaining_row[3]
    execution_id = remaining_row[0]
    monkeypatch.chdir(tmp_path)
    progress = Path("worker_outputs/artifacts.in_progress")
    progress.mkdir(parents=True)
    (progress / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    (Path("worker_outputs") / "attempt_identity.tsv").write_text(
        f"{execution_id}\t123\t4\t1\n",
        encoding="utf-8",
    )
    (Path("worker_outputs") / "worker_exit_status.txt").write_text(
        "2\n",
        encoding="utf-8",
    )
    args = _attempt_args(
        execution_id=execution_id,
        job_path=job_path,
        authorization_path=authorization_path,
        worker_exit_status=2,
    )

    result = builder.build_archive(args)
    assert result["status"] == "passed"
    with tarfile.open(args.output_archive, "r:gz") as archive:
        attempt_receipt = json.load(
            archive.extractfile("worker_attempt_receipt.json")
        )
        names = set(archive.getnames())
    assert attempt_receipt["science_evidence_state"] == (
        "in_progress_science_preserved_unvalidated_v2"
    )
    assert (
        "worker_outputs/artifacts.in_progress/checkpoint.json" in names
    )
