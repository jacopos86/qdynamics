from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "finalize_page12_insertion_comparator_closure_20260813.py"
)
REPAIR_RELATIVE = Path("chtc/paper_i_ra_adapt_repair_20260727")
PACKAGE_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
ACTIVATION_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_"
    "20260812_v1_chtc_activation_v1"
)
ARCHIVE_DIR_RELATIVE = REPAIR_RELATIVE / (
    "retrieved_page12_insertion_comparators_20260813"
)
EVIDENCE_DIR_RELATIVE = REPAIR_RELATIVE / (
    "page12_insertion_comparator_closure_evidence"
)
RECEIPT_DIR_RELATIVE = REPAIR_RELATIVE / (
    "page12_insertion_comparator_closure_receipts"
)
CLUSTER_ID = 9_647_385


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digested(value: dict) -> dict:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _json_bytes(value: dict) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _job(proc: int) -> dict:
    queue = (REPO_ROOT / PACKAGE_RELATIVE / "queue.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    execution_id, job_path, *_ = queue[proc].split("\t")
    value = json.loads(
        (REPO_ROOT / PACKAGE_RELATIVE / job_path).read_text(encoding="utf-8")
    )
    assert value["execution_id"] == execution_id
    return value


def _authorization(run_id: str) -> dict:
    return json.loads(
        (
            REPO_ROOT
            / ACTIVATION_RELATIVE
            / "authorizations"
            / f"{run_id}.json"
        ).read_text(encoding="utf-8")
    )


def _archive_relative(proc: int, run_id: str) -> Path:
    return ARCHIVE_DIR_RELATIVE / f"{run_id}__{CLUSTER_ID}__{proc}.tar.gz"


def _receipt_relative(proc: int, run_id: str) -> Path:
    return RECEIPT_DIR_RELATIVE / (
        f"paper_i_ra_adapt_page12_cluster{CLUSTER_ID}_proc{proc:02d}_"
        f"{run_id}_closure_receipt_20260813.json"
    )


def _identity_relative(proc: int, run_id: str) -> Path:
    return EVIDENCE_DIR_RELATIVE / (
        f"{run_id}__{CLUSTER_ID}__{proc}_remote_archive_identity.json"
    )


def _sealed_archive(
    workspace_root: Path,
    *,
    proc: int = 0,
    add_unbound_file: bool = False,
    drift_worker_artifact: bool = False,
    duplicate_worker_receipt: bool = False,
) -> tuple[Path, dict, dict, dict[str, bytes]]:
    job = _job(proc)
    run_id = job["execution_id"]
    authority = _authorization(run_id)
    expected = {
        role: row["path"]
        for role, row in job["expected_run_artifacts"].items()
    }
    checkpoint = PurePosixPath(expected["checkpoint"])
    ledger_sidecar_payload = _json_bytes({"ledger_checkpoint": "sealed"})
    resume_sidecar_payload = _json_bytes({"verified_resume": "sealed"})
    sidecars = {
        checkpoint.with_name(
            f"{checkpoint.stem}.estimator_call_ledger_checkpoint."
            f"{_sha256(ledger_sidecar_payload)[:16]}.json"
        ).as_posix(): ledger_sidecar_payload,
        checkpoint.with_name(
            f"{checkpoint.stem}.verified_singleton_resume."
            f"{_sha256(resume_sidecar_payload)[:16]}.json"
        ).as_posix(): resume_sidecar_payload,
    }
    files: dict[str, bytes] = {
        expected["checkpoint"]: _json_bytes({"checkpoint": "sealed"}),
        expected["estimator_ledger"]: _json_bytes({"ledger": "sealed"}),
        expected["result"]: _json_bytes({"result": "sealed"}),
        expected["summary"]: _json_bytes(
            {
                "schema": "paper_i_run_summary_v1",
                "method": "ra_adapt",
                "status": "passed",
            }
        ),
        **sidecars,
    }
    output_payloads = {
        role: {
            "path": expected[role],
            "sha256": _sha256(files[expected[role]]),
            "size_bytes": len(files[expected[role]]),
        }
        for role in ("checkpoint", "estimator_ledger", "result", "summary")
    }
    manifest = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": run_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authority["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "target_horizon": job["target_horizon"],
            "comparator_policy": job["comparator_policy"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": output_payloads,
        }
    )
    files[expected["execution_manifest"]] = _json_bytes(manifest)
    artifacts = [
        {
            "path": path,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(files.items())
    ]
    if drift_worker_artifact:
        artifacts[0]["sha256"] = "0" * 64
    worker = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": run_id,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authority["sha256"],
            "execution_manifest_sha256": manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": artifacts,
        }
    )
    archive_members = {
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": _json_bytes(worker),
        **files,
    }
    if add_unbound_file:
        archive_members["unbound.json"] = b"{}\n"
    archive_path = workspace_root / _archive_relative(proc, run_id)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        directories = sorted(
            {
                parent.as_posix()
                for name in archive_members
                for parent in PurePosixPath(name).parents
                if parent.as_posix() != "."
            },
            key=lambda value: (value.count("/"), value),
        )
        for directory in directories:
            info = tarfile.TarInfo(f"./{directory}")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in archive_members.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            import io

            archive.addfile(info, io.BytesIO(payload))
        if duplicate_worker_receipt:
            payload = archive_members["worker_receipt.json"]
            info = tarfile.TarInfo("./worker_receipt.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return archive_path, worker, manifest, files


def _remote_identity(
    workspace_root: Path,
    archive_path: Path,
    *,
    proc: int = 0,
) -> Path:
    job = _job(proc)
    run_id = job["execution_id"]
    payload = archive_path.read_bytes()
    archive_relative = _archive_relative(proc, run_id).as_posix()
    remote_path = (
        "osdf:///chtc/staging/j/jsstrobel/"
        "paper_i_ra_adapt_page12_insertion_comparators_20260812_v1/outputs/"
        f"transfer/{run_id}__{CLUSTER_ID}__{proc}.tar.gz"
    )
    identity = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_page12_insertion_comparator_"
                "remote_archive_identity_v1"
            ),
            "status": "passed_remote_local_size_sha256_match_after_atomic_rename",
            "captured_at_utc": "2026-08-13T10:00:00Z",
            "cluster_id": CLUSTER_ID,
            "proc_id": proc,
            "execution_id": run_id,
            "remote_path": remote_path,
            "local_path": archive_relative,
            "remote_size_bytes": len(payload),
            "local_size_bytes": len(payload),
            "remote_sha256": _sha256(payload),
            "local_sha256": _sha256(payload),
            "gzip_integrity_passed": True,
            "tar_readability_passed": True,
            "atomic_local_rename_completed": True,
            "remote_state": "preserved_after_exact_size_sha256_verified_fetch",
        }
    )
    path = workspace_root / _identity_relative(proc, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(identity))
    return path


def _run_helper(
    workspace_root: Path,
    mode: str,
    *,
    proc: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            HELPER_PATH.as_posix(),
            mode,
            "--proc",
            str(proc),
            "--workspace-root",
            workspace_root.as_posix(),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_preflight_authenticates_fixed_proc_archive_without_writes(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, worker, manifest, files = _sealed_archive(workspace_root)
    identity_path = _remote_identity(workspace_root, archive_path)
    before_archive = archive_path.read_bytes()
    before_identity = identity_path.read_bytes()

    completed = _run_helper(workspace_root, "--preflight")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["schema"] == (
        "paper_i_ra_adapt_page12_insertion_comparator_closure_preflight_v1"
    )
    assert result["status"] == "passed_ready_to_finalize"
    assert result["cluster_id"] == CLUSTER_ID
    assert result["proc_id"] == 0
    assert result["archive"]["regular_member_count"] == 9
    assert result["archive"]["declared_artifact_count"] == 7
    assert result["worker_receipt_canonical_sha256"] == worker["sha256"]
    assert result["execution_manifest_canonical_sha256"] == manifest["sha256"]
    assert result["summary_json"]["path_inside_archive"] in files
    assert result["writes_performed"] is False
    assert result["network_performed"] is False
    assert result["scheduler_mutation_performed"] is False
    assert archive_path.read_bytes() == before_archive
    assert identity_path.read_bytes() == before_identity
    job = _job(0)
    assert not (workspace_root / _receipt_relative(0, job["execution_id"])).exists()


def test_finalize_atomically_mints_reporting_ready_receipt(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, worker, manifest, files = _sealed_archive(workspace_root)
    identity_path = _remote_identity(workspace_root, archive_path)
    before_archive = archive_path.read_bytes()
    before_identity = identity_path.read_bytes()
    job = _job(0)
    run_id = job["execution_id"]

    completed = _run_helper(workspace_root, "--finalize")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed_strict_receipt_atomically_published"
    receipt_relative = _receipt_relative(0, run_id)
    assert result["receipt_path"] == receipt_relative.as_posix()
    receipt_path = workspace_root / receipt_relative
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
    assert receipt["sha256"] == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    assert receipt["schema"] == (
        "paper_i_ra_adapt_page12_insertion_comparator_closure_receipt_v1"
    )
    assert receipt["status"] == (
        "passed_authenticated_page12_insertion_comparator_closure"
    )
    assert receipt["cluster_id"] == CLUSTER_ID
    assert receipt["proc_id"] == 0
    assert receipt["run_id"] == run_id
    assert receipt["regime_id"] == "weak_weak"
    assert receipt["comparator_policy"] == "always_commutation_reduced"
    assert receipt["typed_insertion_kind"] == "always_commutation_reduced"
    assert receipt["runtime_insertion_mode"] == "full_commutation_reduced"
    assert receipt["controller_rounds_completed"] == 50
    assert receipt["package_manifest"]["canonical_sha256"] == (
        "efce225efdc04653e8fca7e34eb3f467d4a6ec594e2130cde4bbea45e3d040e9"
    )
    assert receipt["job"]["canonical_sha256"] == job["sha256"]
    assert receipt["activation_manifest"]["canonical_sha256"] == (
        "9aa36c3362257dfdcd8624bf091adfbaae28edb06e0abadcb8d6b6936533a36d"
    )
    assert receipt["authorization"]["canonical_sha256"] == (
        _authorization(run_id)["sha256"]
    )
    assert receipt["archive"]["path"] == _archive_relative(0, run_id).as_posix()
    assert receipt["archive"]["sha256"] == hashlib.sha256(before_archive).hexdigest()
    inventory = receipt["archive"]["inventory"]
    assert len(inventory) == 9
    assert all(not row["path"].startswith("./") for row in inventory)
    assert {row["path"] for row in inventory} == {
        "worker_exit_status.txt",
        "worker_receipt.json",
        *files,
    }
    assert receipt["worker_receipt"]["canonical_sha256"] == worker["sha256"]
    assert receipt["execution_manifest"]["canonical_sha256"] == manifest["sha256"]
    assert receipt["summary_json"]["path_inside_archive"] in files
    assert all(receipt["authentication_checks"].values())
    assert receipt["network_performed_by_action"] is False
    assert receipt["scheduler_mutation_performed_by_action"] is False
    assert receipt["scientific_execution_performed_by_action"] is False
    assert archive_path.read_bytes() == before_archive
    assert identity_path.read_bytes() == before_identity
    assert list(receipt_path.parent.glob(".proc-receipt.*.tmp")) == []


def test_finalize_rejects_archive_member_unbound_by_worker_without_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _, _, _ = _sealed_archive(
        workspace_root,
        add_unbound_file=True,
    )
    _remote_identity(workspace_root, archive_path)
    run_id = _job(0)["execution_id"]

    completed = _run_helper(workspace_root, "--finalize")

    assert completed.returncode == 2
    assert "archive contains missing or unbound files" in completed.stderr
    assert not (workspace_root / _receipt_relative(0, run_id)).exists()
    receipt_parent = workspace_root / RECEIPT_DIR_RELATIVE
    assert not receipt_parent.exists()


def test_preflight_rejects_worker_declared_hash_drift_without_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _, _, _ = _sealed_archive(
        workspace_root,
        drift_worker_artifact=True,
    )
    _remote_identity(workspace_root, archive_path)
    run_id = _job(0)["execution_id"]

    completed = _run_helper(workspace_root, "--preflight")

    assert completed.returncode == 2
    assert "worker artifact binding drifted" in completed.stderr
    assert not (workspace_root / _receipt_relative(0, run_id)).exists()


def test_proc_six_is_the_fixed_weak_weak_append_only_row(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _, _, _ = _sealed_archive(workspace_root, proc=6)
    _remote_identity(workspace_root, archive_path, proc=6)

    completed = _run_helper(workspace_root, "--preflight", proc=6)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["proc_id"] == 6
    assert result["regime_id"] == "weak_weak"
    assert result["comparator_policy"] == "append_only"
    assert result["run_id"].endswith("_append_only")


def test_finalize_rejects_redigested_wrong_remote_identity_without_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _, _, _ = _sealed_archive(workspace_root)
    identity_path = _remote_identity(workspace_root, archive_path)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["remote_sha256"] = "0" * 64
    identity_path.write_bytes(_json_bytes(_digested(identity)))
    run_id = _job(0)["execution_id"]

    completed = _run_helper(workspace_root, "--finalize")

    assert completed.returncode == 2
    assert "remote/local archive identity evidence drifted" in completed.stderr
    assert not (workspace_root / _receipt_relative(0, run_id)).exists()


def test_preflight_rejects_duplicate_tar_member_without_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    archive_path, _, _, _ = _sealed_archive(
        workspace_root,
        duplicate_worker_receipt=True,
    )
    _remote_identity(workspace_root, archive_path)
    run_id = _job(0)["execution_id"]

    completed = _run_helper(workspace_root, "--preflight")

    assert completed.returncode == 2
    assert "duplicate archive member: worker_receipt.json" in completed.stderr
    assert not (workspace_root / _receipt_relative(0, run_id)).exists()


def test_finalizer_source_is_network_scheduler_and_deletion_inert() -> None:
    source = HELPER_PATH.read_text(encoding="utf-8")

    assert "import subprocess" not in source
    assert "import socket" not in source
    assert "requests" not in source
    assert "paramiko" not in source
    assert "condor_" not in source
    assert "ssh " not in source.lower()
    assert ".unlink(" not in source
    assert "os.remove(" not in source
    assert "shutil.rmtree(" not in source
