from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "finalize_page16_sw_always_closure_20260813.py"
)
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc"
)
EXECUTION_ID = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced"
)
ARCHIVE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_page16_insertion_comparators_20260812/"
    "strong_weak_u8_always__9647386__1.tar.gz"
)
RECEIPT_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_cluster9647386_sw_always_"
    "remote_materialization_exclusion_receipt_20260813.json"
)
REMOTE_ARCHIVE_PATH = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_adapt_page16_insertion_comparators_20260812_v1/"
    "outputs/transfer/"
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced__9647386__1.tar.gz"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _write_digested(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _digested(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload) + b"\n")
    return payload


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name=name)
    member.size = len(payload)
    member.mtime = 0
    member.mode = 0o600
    archive.addfile(member, io.BytesIO(payload))


def _sealed_archive(
    workspace_root: Path,
    *,
    sidecar_content_address_drift: bool = False,
    add_unbound_file: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    job_path = PACKAGE_DIR / "jobs" / f"{EXECUTION_ID}.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    expected = job["expected_run_artifacts"]
    checkpoint_path = PurePosixPath(expected["checkpoint"]["path"])
    ledger_sidecar_payload = _canonical_bytes(
        {
            "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v2",
            "fixture_role": "estimator_call_ledger_checkpoint",
            "controller_round": 50,
        }
    ) + b"\n"
    ledger_sidecar_sha256 = hashlib.sha256(ledger_sidecar_payload).hexdigest()
    ledger_digest_prefix = ledger_sidecar_sha256[:16]
    if sidecar_content_address_drift:
        ledger_digest_prefix = (
            "0" * 16 if ledger_digest_prefix != "0" * 16 else "1" * 16
        )
    ledger_sidecar_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.estimator_call_ledger_checkpoint."
        f"{ledger_digest_prefix}.json"
    ).as_posix()
    ledger_pointer = {
        "schema": "paper_i_estimator_call_ledger_checkpoint_pointer_v2",
        "enabled": True,
        "status": "complete",
        "path": PurePosixPath(ledger_sidecar_path).name,
        "sha256": ledger_sidecar_sha256,
    }
    resume_sidecar_payload = _canonical_bytes(
        {
            "schema": "static_adapt_signed_active_prefix_resume_sidecar_v2",
            "fixture_role": "verified_singleton_resume",
            "controller_round": 50,
        }
    ) + b"\n"
    resume_sidecar_sha256 = hashlib.sha256(resume_sidecar_payload).hexdigest()
    resume_sidecar_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}.verified_singleton_resume."
        f"{resume_sidecar_sha256[:16]}.json"
    ).as_posix()
    resume_pointer = {
        "schema": "static_adapt_verified_singleton_resume_sidecar_pointer_v1",
        "enabled": True,
        "status": "complete",
        "path": PurePosixPath(resume_sidecar_path).name,
        "sha256": resume_sidecar_sha256,
    }
    payloads = {
        "checkpoint": _canonical_bytes(
            {
                "fixture_role": "checkpoint",
                "controller_round": 50,
                "checkpoint": {
                    "estimator_call_ledger_checkpoint": ledger_pointer,
                },
                "adapt_vqe": {
                    "estimator_call_ledger_checkpoint": ledger_pointer,
                    "verified_singleton_resume_sidecar": resume_pointer,
                },
            }
        )
        + b"\n",
        **{
            role: _canonical_bytes(
                {"fixture_role": role, "controller_round": 50}
            )
            + b"\n"
            for role in ("estimator_ledger", "result", "summary")
        },
    }
    output_payloads = {
        role: {
            "path": expected[role]["path"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for role, payload in payloads.items()
    }
    manifest = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_page16_macro_phase23_qiskit_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": "a" * 64,
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "target_horizon": 50,
            "comparator_policy": "always_commutation_reduced",
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": output_payloads,
        }
    )
    manifest_payload = _canonical_bytes(manifest) + b"\n"
    artifact_payloads = {
        **{
            expected[role]["path"]: payload
            for role, payload in payloads.items()
        },
        expected["execution_manifest"]["path"]: manifest_payload,
        ledger_sidecar_path: ledger_sidecar_payload,
        resume_sidecar_path: resume_sidecar_payload,
    }
    worker = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_page16_macro_phase23_qiskit_"
                "worker_receipt_v1"
            ),
            "status": "passed",
            "package_id": job["package_id"],
            "campaign_id": job["campaign_id"],
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": "a" * 64,
            "execution_manifest_sha256": manifest["sha256"],
            "controller_rounds_completed": 50,
            "fresh_start": True,
            "artifacts": [
                {
                    "path": path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for path, payload in sorted(artifact_payloads.items())
            ],
        }
    )
    archive_path = workspace_root / ARCHIVE_RELATIVE
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        root = tarfile.TarInfo(name=".")
        root.type = tarfile.DIRTYPE
        root.mtime = 0
        root.mode = 0o755
        archive.addfile(root)
        _add_tar_bytes(archive, "./worker_exit_status.txt", b"0\n")
        _add_tar_bytes(
            archive,
            "./worker_receipt.json",
            _canonical_bytes(worker) + b"\n",
        )
        for path, payload in sorted(artifact_payloads.items()):
            _add_tar_bytes(archive, f"./{path}", payload)
        if add_unbound_file:
            _add_tar_bytes(archive, "./unbound.json", b"{}\n")
    return archive_path, worker, manifest


def _remote_identity(evidence_dir: Path, archive_path: Path) -> Path:
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return_path = evidence_dir / "page16_sw_always_remote_archive_identity.json"
    _write_digested(
        return_path,
        {
            "schema": "paper_i_page16_sw_always_remote_archive_identity_v1",
            "status": "passed_remote_local_size_sha256_match_after_atomic_rename",
            "captured_at_utc": "2026-08-13T05:00:00Z",
            "remote_path": REMOTE_ARCHIVE_PATH,
            "local_path": ARCHIVE_RELATIVE.as_posix(),
            "remote_size_bytes": archive_path.stat().st_size,
            "local_size_bytes": archive_path.stat().st_size,
            "remote_sha256": digest,
            "local_sha256": digest,
            "gzip_integrity_passed": True,
            "tar_readability_passed": True,
            "atomic_local_rename_completed": True,
            "remote_state": "preserved_after_exact_size_sha256_verified_fetch",
        },
    )
    return return_path


AUTHENTICATION = {
    "authenticated_remote_query": True,
    "kind": "interactive_ssh_duo_condor_q_snapshot_v1",
    "source_host": "ap2001.chtc.wisc.edu",
}
QUEUE_COMMAND = (
    "condor_q 9647386 -json -attributes "
    "ClusterId,ProcId,JobStatus,NumJobStarts"
)
FACTORY_COMMAND = (
    "condor_q -factory 9647386 -json -attributes "
    "ClusterId,TotalSubmitProcs,JobMaterializeLimit,JobMaterializeMaxIdle,"
    "JobMaterializeNextProcId,JobMaterializePaused"
)
HISTORY_COMMAND = (
    "condor_history 9647386 -limit 20 -json -attributes "
    "ClusterId,ProcId,JobStatus,ExitCode,NumJobStarts,CompletionDate"
)


def _history_rows() -> list[dict[str, int]]:
    return [
        {
            "ClusterId": 9_647_386,
            "ProcId": proc_id,
            "JobStatus": 4,
            "ExitCode": 0,
            "NumJobStarts": 1,
            "CompletionDate": 1_786_596_900 + proc_id,
        }
        for proc_id in (0, 1)
    ]


def _closure_evidence(
    evidence_dir: Path,
    *,
    retained_paused_factory: bool = True,
) -> None:
    before_rows = _history_rows()
    _write_digested(
        evidence_dir / "page16_cluster9647386_before_retirement.json",
        {
            "schema": (
                "paper_i_page16_sw_always_factory_before_retirement_"
                "snapshot_v1"
            ),
            "status": "passed_authenticated_paused_factory_before_retirement",
            "captured_at_utc": "2026-08-13T05:10:00Z",
            "authentication": AUTHENTICATION,
            "cluster_id": 9_647_386,
            "queue_query": {"command": QUEUE_COMMAND, "rows": []},
            "factory_query": {
                "command": FACTORY_COMMAND,
                "rows": [
                    {
                        "ClusterId": 9_647_386,
                        "TotalSubmitProcs": 11,
                        "JobMaterializeLimit": 1,
                        "JobMaterializeMaxIdle": 0,
                        "JobMaterializeNextProcId": 2,
                        "JobMaterializePaused": 1,
                    }
                ],
            },
            "history_query": {
                "command": HISTORY_COMMAND,
                "rows": before_rows,
            },
        },
    )
    _write_digested(
        evidence_dir / "page16_cluster9647386_acknowledged_removal_attempts.json",
        {
            "schema": (
                "paper_i_page16_sw_always_acknowledged_removal_attempts_v1"
            ),
            "status": (
                "passed_authenticated_at_least_one_acknowledged_condor_rm"
            ),
            "captured_at_utc": "2026-08-13T05:11:00Z",
            "authentication": AUTHENTICATION,
            "cluster_id": 9_647_386,
            "attempts": [
                {
                    "command": "condor_rm 9647386",
                    "started_at_utc": "2026-08-13T05:10:30Z",
                    "acknowledged_at_utc": "2026-08-13T05:10:31Z",
                    "exit_code": 0,
                    "acknowledged": True,
                    "acknowledgement_text": (
                        "All jobs in cluster 9647386 have been marked for removal"
                    ),
                }
            ],
        },
    )
    _write_digested(
        evidence_dir
        / "page16_cluster9647386_after_materialization_exclusion.json",
        {
            "schema": (
                "paper_i_page16_sw_always_remote_materialization_exclusion_"
                "after_snapshot_v2"
            ),
            "status": (
                "passed_authenticated_remote_materialization_excluded_"
                "after_removal_attempt"
            ),
            "captured_at_utc": "2026-08-13T05:12:00Z",
            "authentication": AUTHENTICATION,
            "cluster_id": 9_647_386,
            "queue_query": {"command": QUEUE_COMMAND, "rows": []},
            "factory_query": {
                "command": FACTORY_COMMAND,
                "rows": (
                    [
                        {
                            "ClusterId": 9_647_386,
                            "TotalSubmitProcs": 11,
                            "JobMaterializeLimit": 2,
                            "JobMaterializeMaxIdle": 0,
                            "JobMaterializeNextProcId": 2,
                            "JobMaterializePaused": 1,
                        }
                    ]
                    if retained_paused_factory
                    else []
                ),
            },
            "history_query": {
                "command": HISTORY_COMMAND,
                "rows": _history_rows(),
            },
        },
    )
    _write_digested(
        evidence_dir / "page16_cluster9647386_history.json",
        {
            "schema": "paper_i_page16_sw_always_cluster_history_snapshot_v2",
            "status": "passed_authenticated_only_procs_0_1_completed",
            "captured_at_utc": "2026-08-13T05:13:00Z",
            "authentication": AUTHENTICATION,
            "cluster_id": 9_647_386,
            "queried_proc_ids": list(range(11)),
            "history_query": {
                "command": HISTORY_COMMAND,
                "rows": _history_rows(),
            },
        },
    )


def _run_helper(
    workspace_root: Path,
    evidence_dir: Path,
    mode: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            HELPER_PATH.as_posix(),
            mode,
            "--workspace-root",
            workspace_root.as_posix(),
            "--evidence-dir",
            evidence_dir.as_posix(),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_preflight_deeply_authenticates_archive_and_prints_exclusion_plan(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, worker, manifest = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    before_archive = archive_path.read_bytes()

    completed = _run_helper(workspace_root, evidence_dir, "--preflight")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == (
        "passed_archive_ready_for_user_mediated_remote_materialization_exclusion"
    )
    assert result["archive_closure"] == {
        "archive_sha256": hashlib.sha256(before_archive).hexdigest(),
        "archive_size_bytes": len(before_archive),
        "artifact_count": 7,
        "all_declared_payload_hashes_verified": True,
        "execution_manifest_canonical_sha256": manifest["sha256"],
        "unbound_file_count": 0,
        "worker_exit_status": 0,
        "worker_receipt_canonical_sha256": worker["sha256"],
    }
    plan = result["remote_materialization_exclusion_plan"]
    assert plan["target_cluster_id"] == 9_647_386
    assert plan["removal_command"] == "condor_rm 9647386"
    assert plan["latent_proc_ids_that_must_never_materialize"] == list(
        range(2, 11)
    )
    assert plan["helper_executes_commands"] is False
    assert plan["requires_interactive_ssh_duo"] is True
    evidence_contract = result["evidence_contract"]
    assert evidence_contract["authentication"] == AUTHENTICATION
    assert evidence_contract["required_files"] == [
        {
            "capture_order": 0,
            "filename": "page16_sw_always_remote_archive_identity.json",
            "schema": "paper_i_page16_sw_always_remote_archive_identity_v1",
            "status": "passed_remote_local_size_sha256_match_after_atomic_rename",
        },
        {
            "capture_order": 1,
            "filename": "page16_cluster9647386_before_retirement.json",
            "schema": (
                "paper_i_page16_sw_always_factory_before_retirement_"
                "snapshot_v1"
            ),
            "status": "passed_authenticated_paused_factory_before_retirement",
        },
        {
            "capture_order": 2,
            "filename": (
                "page16_cluster9647386_acknowledged_removal_attempts.json"
            ),
            "schema": (
                "paper_i_page16_sw_always_acknowledged_removal_attempts_v1"
            ),
            "status": (
                "passed_authenticated_at_least_one_acknowledged_condor_rm"
            ),
        },
        {
            "capture_order": 3,
            "filename": (
                "page16_cluster9647386_after_materialization_exclusion.json"
            ),
            "schema": (
                "paper_i_page16_sw_always_remote_materialization_exclusion_"
                "after_snapshot_v2"
            ),
            "status": (
                "passed_authenticated_remote_materialization_excluded_"
                "after_removal_attempt"
            ),
        },
        {
            "capture_order": 4,
            "filename": "page16_cluster9647386_history.json",
            "schema": "paper_i_page16_sw_always_cluster_history_snapshot_v2",
            "status": "passed_authenticated_only_procs_0_1_completed",
        },
    ]
    assert evidence_contract["publication_rule"] == (
        "archive_and_all_evidence_first_atomic_strict_receipt_last"
    )
    assert result["writes_performed"] is False
    assert result["scheduler_mutation_performed"] is False
    assert archive_path.read_bytes() == before_archive
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_finalize_atomically_mints_the_strict_consumer_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, worker, manifest = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir)
    before_archive = archive_path.read_bytes()

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "passed_strict_receipt_atomically_published"
    assert result["scheduler_mutation_performed"] is False
    assert result["scientific_execution_performed"] is False
    receipt_path = workspace_root / RECEIPT_RELATIVE
    assert result["receipt_path"] == RECEIPT_RELATIVE.as_posix()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in receipt.items() if key != "sha256"}
    assert receipt["sha256"] == hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    assert receipt["schema"] == (
        "paper_i_ra_adapt_page16_sw_always_"
        "remote_materialization_exclusion_receipt_v2"
    )
    assert receipt["status"] == (
        "passed_sw_always_k50_closed_remote_materialization_excluded"
    )
    completed_cell = receipt["completed_remote_cell"]
    assert completed_cell["archive"] == {
        "path": ARCHIVE_RELATIVE.as_posix(),
        "remote_path": REMOTE_ARCHIVE_PATH,
        "remote_size_bytes": len(before_archive),
        "local_size_bytes": len(before_archive),
        "size_bytes": len(before_archive),
        "remote_sha256": hashlib.sha256(before_archive).hexdigest(),
        "local_sha256": hashlib.sha256(before_archive).hexdigest(),
        "sha256": hashlib.sha256(before_archive).hexdigest(),
    }
    assert completed_cell["worker_receipt"] == {
        "path_inside_archive": "worker_receipt.json",
        "canonical_sha256": worker["sha256"],
        "schema": worker["schema"],
        "status": "passed",
    }
    assert completed_cell["execution_manifest"] == {
        "path_inside_archive": f"runs/{EXECUTION_ID}/execution_manifest.json",
        "canonical_sha256": manifest["sha256"],
    }
    assert completed_cell["history"] == {
        "cluster_id": 9_647_386,
        "proc_id": 1,
        "job_status": 4,
        "exit_code": 0,
        "num_job_starts": 1,
        "completion_date_epoch": 1_786_596_901,
    }
    exclusion = receipt["remote_materialization_exclusion"]
    assert exclusion["outcome"] == (
        "factory_retained_paused_at_completed_prefix_after_acknowledged_removal"
    )
    assert exclusion["removal_command"] == "condor_rm 9647386"
    assert exclusion["removal_attempts_authenticated"] is True
    assert exclusion["before_snapshot"] == {
        "job_materialize_paused": 1,
        "job_materialize_next_proc_id": 2,
        "materialized_proc_ids": [],
        "history_completed_proc_ids": [0, 1],
    }
    assert exclusion["after_snapshot"] == {
        "cluster_present_in_queue": False,
        "factory_present": True,
        "factory_materialization_paused": True,
        "job_materialize_limit": 2,
        "job_materialize_max_idle": 0,
        "job_materialize_next_proc_id": 2,
        "history_completed_proc_ids": [0, 1],
    }
    assert exclusion["latent_proc_ids_never_materialized"] == list(range(2, 11))
    assert exclusion["queue_cluster_absent"] is True
    assert exclusion["remote_materialization_excluded"] is True
    assert receipt["authentication"] == AUTHENTICATION
    assert receipt["scientific_execution_performed_by_action"] is False
    assert archive_path.read_bytes() == before_archive
    assert list(receipt_path.parent.glob(f".{receipt_path.name}.*.tmp")) == []


def test_finalize_accepts_factory_absent_after_acknowledged_removal(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir, retained_paused_factory=False)

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(
        (workspace_root / RECEIPT_RELATIVE).read_text(encoding="utf-8")
    )
    exclusion = receipt["remote_materialization_exclusion"]
    assert exclusion["outcome"] == (
        "factory_absent_after_acknowledged_removal"
    )
    assert exclusion["after_snapshot"] == {
        "cluster_present_in_queue": False,
        "factory_present": False,
        "factory_materialization_paused": None,
        "job_materialize_limit": None,
        "job_materialize_max_idle": None,
        "job_materialize_next_proc_id": None,
        "history_completed_proc_ids": [0, 1],
    }


def test_preflight_rejects_checkpoint_sidecar_with_drifted_content_address(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(
        workspace_root,
        sidecar_content_address_drift=True,
    )
    _remote_identity(evidence_dir, archive_path)

    completed = _run_helper(workspace_root, evidence_dir, "--preflight")

    assert completed.returncode == 2
    assert (
        "worker checkpoint sidecar content address drifted: "
        "estimator_call_ledger_checkpoint"
    ) in completed.stderr
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_preflight_rejects_an_archive_file_unbound_by_the_worker_receipt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(
        workspace_root,
        add_unbound_file=True,
    )
    _remote_identity(evidence_dir, archive_path)

    completed = _run_helper(workspace_root, evidence_dir, "--preflight")

    assert completed.returncode == 2
    assert "archive contains missing or unbound files" in completed.stderr
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_finalize_rejects_nonempty_before_queue_without_publishing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir)
    before_path = evidence_dir / "page16_cluster9647386_before_retirement.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    before["queue_query"]["rows"] = [
        {
            "ClusterId": 9_647_386,
            "ProcId": 1,
            "JobStatus": 4,
            "NumJobStarts": 1,
        }
    ]
    _write_digested(before_path, before)

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 2
    assert "before-retirement queue must be empty" in completed.stderr
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_finalize_rejects_after_history_with_a_latent_proc_without_publishing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir)
    after_path = (
        evidence_dir
        / "page16_cluster9647386_after_materialization_exclusion.json"
    )
    after = json.loads(after_path.read_text(encoding="utf-8"))
    after["history_query"]["rows"].append(
        {
            "ClusterId": 9_647_386,
            "ProcId": 2,
            "JobStatus": 4,
            "ExitCode": 0,
            "NumJobStarts": 1,
            "CompletionDate": 1_786_596_902,
        }
    )
    _write_digested(after_path, after)

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 2
    assert "after-exclusion history completed-proc identity drifted" in (
        completed.stderr
    )
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_finalize_rejects_retained_factory_that_is_not_paused(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir)
    after_path = (
        evidence_dir
        / "page16_cluster9647386_after_materialization_exclusion.json"
    )
    after = json.loads(after_path.read_text(encoding="utf-8"))
    after["factory_query"]["rows"][0]["JobMaterializePaused"] = 0
    _write_digested(after_path, after)

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 2
    assert "retained factory materialization exclusion drifted" in completed.stderr
    assert not (workspace_root / RECEIPT_RELATIVE).exists()


def test_finalize_rejects_unacknowledged_removal_attempt(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    evidence_dir = tmp_path / "evidence"
    archive_path, _, _ = _sealed_archive(workspace_root)
    _remote_identity(evidence_dir, archive_path)
    _closure_evidence(evidence_dir)
    attempts_path = (
        evidence_dir
        / "page16_cluster9647386_acknowledged_removal_attempts.json"
    )
    attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
    attempts["attempts"][0]["acknowledged"] = False
    _write_digested(attempts_path, attempts)

    completed = _run_helper(workspace_root, evidence_dir, "--finalize")

    assert completed.returncode == 2
    assert "acknowledged removal attempt drifted" in completed.stderr
    assert not (workspace_root / RECEIPT_RELATIVE).exists()
