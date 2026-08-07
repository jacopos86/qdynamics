from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import ModuleType
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
UTILITY_PATH = REPAIR_ROOT / "validate_append_r70_attempt_archive.py"
PACKAGE_DIR = (
    REPAIR_ROOT
    / "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc"
)
ACTIVATION_DIR = (
    REPAIR_ROOT
    / "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc_activation_ordinary_held_v1"
)
SUBMISSION_RECEIPT_PATH = REPAIR_ROOT / (
    "paper_i_append_adapt_stationary_core12_r70_fresh_"
    "20260731_v1_chtc_activation_ordinary_held_v1_submission_receipt.json"
)
EXECUTION_ID = "r70_fresh__strong_weak_u8__nph3__append_macro"
CLUSTER_ID = 9_398_375
PROC_ID = 4


def _load_utility() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_append_r70_attempt_archive",
        UTILITY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(scope="module")
def utility() -> ModuleType:
    return _load_utility()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _json_bytes(utility: ModuleType, payload: dict[str, Any]) -> bytes:
    return utility.canonical_json_bytes(payload) + b"\n"


def _sealed_json_bytes(
    utility: ModuleType,
    payload: dict[str, Any],
) -> bytes:
    return _json_bytes(utility, utility.digested(payload))


@dataclass
class SyntheticAppendAttempt:
    archive_path: Path
    expected: Any
    members: list[tuple[str, bytes, bytes]]

    def write(self) -> None:
        with tarfile.open(self.archive_path, "w:gz") as archive:
            for name, payload, member_type in self.members:
                info = tarfile.TarInfo(name)
                info.type = member_type
                if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    archive.addfile(info)

    def replace_member(self, name: str, payload: bytes) -> None:
        self.members = [
            (
                member_name,
                payload if member_name == name else member_payload,
                member_type,
            )
            for member_name, member_payload, member_type in self.members
        ]
        self.write()


def _build_attempt(
    tmp_path: Path,
    utility: ModuleType,
    *,
    summary_rounds: int = 70,
    worker_receipt_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> SyntheticAppendAttempt:
    package_path = PACKAGE_DIR / "package_manifest.json"
    job_path = PACKAGE_DIR / f"jobs/{EXECUTION_ID}.json"
    authorization_path = ACTIVATION_DIR / "execution_authorization.json"
    activation_path = ACTIVATION_DIR / "activation_manifest.json"
    job = _json(job_path)
    authorization = _json(authorization_path)
    protocol_sha256 = job["derived_protocol_sha256"]

    checkpoint_bytes = _sealed_json_bytes(
        utility,
        {
            "schema": "paper_i_append_adapt_reconstruction_checkpoint_v1",
            "continuation_boundary": "authenticated_reconstruction_only_v1",
            "public_resume_execution_supported": False,
            "reconstruction_fields_complete": True,
            "fresh_start_execution": True,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "execution_id": EXECUTION_ID,
            "protocol_sha256": protocol_sha256,
            "controller_rounds_completed": 70,
        },
    )
    estimator_ledger_bytes = b'{"schema":"estimator_call_ledger_v1"}\n'
    result_bytes = b'{"schema":"synthetic_append_result_for_archive_test"}\n'
    summary_bytes = _json_bytes(
        utility,
        {
            "schema": "paper_i_append_run_summary_v1",
            "protocol_sha256": protocol_sha256,
            "protocol_horizon": 70,
            "controller_rounds_completed": summary_rounds,
            "stop_reason": "maximum_controller_rounds",
            "accepted_history": [
                {"controller_round": index + 1}
                for index in range(summary_rounds)
            ],
            "accepted_operator_labels": [
                f"operator-{index}" for index in range(summary_rounds)
            ],
            "accepted_generator_identities": [
                f"generator-{index}" for index in range(summary_rounds)
            ],
        },
    )
    preliminary = {
        "checkpoint": checkpoint_bytes,
        "estimator_ledger": estimator_ledger_bytes,
        "result": result_bytes,
        "summary": summary_bytes,
    }
    execution_manifest_bytes = _sealed_json_bytes(
        utility,
        {
            "schema": (
                "paper_i_append_adapt_stationary_core_r70_"
                "execution_manifest_v1"
            ),
            "status": "passed",
            "package_id": utility.PACKAGE_ID,
            "campaign_id": utility.CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "source_execution_id": job["source_execution_id"],
            "job_spec_sha256": job["sha256"],
            "authorization_sha256": authorization["sha256"],
            "source_protocol_sha256": job["source_protocol"][
                "canonical_sha256"
            ],
            "derived_protocol_sha256": protocol_sha256,
            "source_horizon": 50,
            "target_horizon": 70,
            "controller_round_origin": 0,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "completed_utc": "2026-07-31T18:00:00+00:00",
            "output_payloads": {
                role: {
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
                for role, payload in sorted(preliminary.items())
            },
        },
    )
    artifacts = {
        **preliminary,
        "execution_manifest": execution_manifest_bytes,
    }
    worker_receipt_body = {
        "schema": (
            "paper_i_append_adapt_stationary_core_r70_worker_receipt_v1"
        ),
        "status": "passed",
        "package_id": utility.PACKAGE_ID,
        "campaign_id": utility.CAMPAIGN_ID,
        "execution_id": EXECUTION_ID,
        "job_spec_sha256": job["sha256"],
        "authorization_sha256": authorization["sha256"],
        "derived_protocol_sha256": protocol_sha256,
        "fresh_start": True,
        "resume_claimed": False,
        "artifacts": [
            {
                "role": role,
                "path": f"{role}.json",
                "declared_canonical_path": job["artifact_paths"][role],
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
            for role, payload in sorted(artifacts.items())
        ],
    }
    if worker_receipt_mutator is not None:
        worker_receipt_mutator(worker_receipt_body)
    worker_receipt_bytes = _sealed_json_bytes(utility, worker_receipt_body)

    attempt_ordinal = 1
    worker_files = {
        "attempt_identity.tsv": (
            f"{EXECUTION_ID}\t{CLUSTER_ID}\t{PROC_ID}\t{attempt_ordinal}\n"
        ).encode("utf-8"),
        "payload/checkpoint.json": checkpoint_bytes,
        "payload/estimator_ledger.json": estimator_ledger_bytes,
        "payload/execution_manifest.json": execution_manifest_bytes,
        "payload/result.json": result_bytes,
        "payload/summary.json": summary_bytes,
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": worker_receipt_bytes,
    }
    attempt_receipt_bytes = _sealed_json_bytes(
        utility,
        {
            "schema": (
                "paper_i_append_adapt_stationary_core_r70_"
                "worker_attempt_v1"
            ),
            "execution_id": EXECUTION_ID,
            "cluster_id": CLUSTER_ID,
            "proc_id": PROC_ID,
            "attempt_ordinal": attempt_ordinal,
            "worker_exit_status": 0,
            "job_file_sha256": _sha256(job_path.read_bytes()),
            "authorization_file_sha256": _sha256(
                authorization_path.read_bytes()
            ),
            "activation_manifest_file_sha256": _sha256(
                activation_path.read_bytes()
            ),
            "source_archive_sha256": utility.SOURCE_ARCHIVE_SHA256,
            "image_sha256": utility.IMAGE_SHA256,
            "worker_files": [
                {
                    "path": path,
                    "sha256": _sha256(payload),
                    "size_bytes": len(payload),
                }
                for path, payload in sorted(worker_files.items())
            ],
        },
    )
    members = [
        *[
            (f"worker_outputs/{path}", payload, tarfile.REGTYPE)
            for path, payload in sorted(worker_files.items())
        ],
        ("authority/job.json", job_path.read_bytes(), tarfile.REGTYPE),
        (
            "authority/execution_authorization.json",
            authorization_path.read_bytes(),
            tarfile.REGTYPE,
        ),
        (
            "authority/activation_manifest.json",
            activation_path.read_bytes(),
            tarfile.REGTYPE,
        ),
        (
            "worker_attempt_receipt.json",
            attempt_receipt_bytes,
            tarfile.REGTYPE,
        ),
    ]
    archive_path = tmp_path / (
        f"{EXECUTION_ID}__cluster_{CLUSTER_ID}__proc_{PROC_ID}.tar.gz"
    )
    synthetic = SyntheticAppendAttempt(
        archive_path=archive_path,
        expected=utility.ExpectedAppendAttempt(
            execution_id=EXECUTION_ID,
            cluster_id=CLUSTER_ID,
            proc_id=PROC_ID,
            package_manifest_path=package_path,
            job_path=job_path,
            authorization_path=authorization_path,
            activation_manifest_path=activation_path,
        ),
        members=members,
    )
    synthetic.write()
    return synthetic


def _remote_observation(
    synthetic: SyntheticAppendAttempt,
    utility: ModuleType,
) -> Any:
    payload = synthetic.archive_path.read_bytes()
    return utility.RemoteRetrievalObservation(
        receipt_created_utc="2026-07-31T18:12:00Z",
        remote_identity_observed_utc="2026-07-31T18:00:00Z",
        retrieved_utc="2026-07-31T18:10:00Z",
        remote_archive_sha256=_sha256(payload),
        remote_archive_size_bytes=len(payload),
        quota_observed_utc="2026-07-31T18:11:00Z",
        quota_home_used_gib=35.0,
        quota_home_soft_limit_gib=40.0,
        quota_home_hard_limit_gib=50.0,
    )


def test_valid_archive_authenticates_authority_and_worker_declarations(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_append_attempt_archive(
        synthetic.archive_path,
        synthetic.expected,
    )

    assert validation["status"] == "passed"
    assert validation["execution_id"] == EXECUTION_ID
    assert validation["cluster_id"] == CLUSTER_ID
    assert validation["proc_id"] == PROC_ID
    expected_member_payload = sum(
        len(payload)
        for _, payload, member_type in synthetic.members
        if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}
    )
    assert validation["member_validation"] == {
        "gzip_and_full_tar_scan_passed": True,
        "safe_unique_regular_only_member_closure_passed": True,
        "worker_inventory_hash_size_closure_passed": True,
        "authority_byte_identity_passed": True,
        "worker_declared_fresh70_crosslink_checks_passed": True,
        "member_count": 12,
        "worker_file_count": 8,
        "total_member_payload_bytes": expected_member_payload,
        "total_uncompressed_bytes": len(
            gzip.decompress(synthetic.archive_path.read_bytes())
        ),
        "activated_row_uncompressed_limit_bytes": 20_480 * 1024 * 1024,
    }
    assert validation["validation_scope"]["scientific_payload_semantics"] == (
        "worker_declared_crosslinks_checked_not_semantically_replayed"
    )
    assert validation["validation_scope"]["scheduler_terminal_history"] == (
        "not_validated"
    )
    assert not (tmp_path / "worker_outputs").exists()

    receipt = utility.build_retrieval_receipt(
        validation=validation,
        expected=synthetic.expected,
        submission_receipt_path=SUBMISSION_RECEIPT_PATH,
        remote=_remote_observation(synthetic, utility),
    )
    utility.verify_self_digest(receipt, label="retrieval receipt")
    assert receipt["retrieval_classification"] == (
        "remote_local_identity_matched_authority_and_inventory_closed"
    )
    assert receipt["receipt_scope"]["scientific_payload_semantics"] == (
        "worker_declared_not_semantically_replayed"
    )
    assert receipt["receipt_scope"]["release_operation"] == (
        "not_authenticated"
    )
    assert receipt["execution"]["fresh_start"] is True
    assert receipt["execution"]["target_horizon"] == 70
    assert "release" not in receipt
    assert "local_final_rename_completed" not in receipt["retrieval"]
    assert receipt["retrieval"]["remote_archive_path"].endswith(
        synthetic.archive_path.name
    )

    output = tmp_path / "retrieval.json"
    utility.write_new_receipt(output, receipt)
    assert _json(output)["sha256"] == receipt["sha256"]
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Refusing to overwrite",
    ):
        utility.write_new_receipt(output, receipt)


def test_rejects_archived_job_bytes_that_differ_from_sealed_job(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    archived_job = next(
        payload
        for name, payload, _ in synthetic.members
        if name == "authority/job.json"
    )
    synthetic.replace_member("authority/job.json", archived_job + b" ")
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Archived authority bytes",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_worker_payload_tamper_not_closed_by_attempt_inventory(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    synthetic.replace_member(
        "worker_outputs/payload/result.json",
        b'{"tampered":true}\n',
    )
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Worker inventory hash/size closure",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_proc_to_activation_queue_mismatch(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    wrong_expected = replace(synthetic.expected, proc_id=PROC_ID + 1)
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Proc-to-job activation binding drifted",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            wrong_expected,
        )


def test_rejects_worker_receipt_artifact_provenance_mismatch(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    def tamper(receipt: dict[str, Any]) -> None:
        receipt["artifacts"][0]["declared_canonical_path"] = (
            "raw_outputs/foreign/result.json"
        )

    synthetic = _build_attempt(
        tmp_path,
        utility,
        worker_receipt_mutator=tamper,
    )
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Worker artifact binding drifted",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_consistently_rehashed_non70_summary(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility, summary_rounds=69)
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="summary horizon closure drifted",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_nonregular_archive_member(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    synthetic.members[0] = (
        synthetic.members[0][0],
        b"",
        tarfile.SYMTYPE,
    )
    synthetic.write()
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Non-regular tar member",
    ):
        utility.validate_append_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_uncompressed_budget_rejects_member_beyond_activated_row_disk(
    utility: ModuleType,
) -> None:
    limit = 20_480 * 1024 * 1024
    assert utility._bounded_uncompressed_total(
        current_bytes=limit - 1,
        member_bytes=1,
        limit_bytes=limit,
    ) == limit
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="uncompressed size budget",
    ):
        utility._bounded_uncompressed_total(
            current_bytes=limit,
            member_bytes=1,
            limit_bytes=limit,
        )


def test_decompressed_budget_counts_concatenated_zero_gzip_payload(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    tar_payload = gzip.decompress(synthetic.archive_path.read_bytes())
    concatenated = (
        gzip.compress(tar_payload, mtime=0)
        + gzip.compress(b"\0" * (2 * 1024 * 1024), mtime=0)
    )
    with gzip.GzipFile(fileobj=io.BytesIO(concatenated), mode="rb") as stream:
        bounded = utility._BoundedDecompressedReader(
            stream,
            limit_bytes=len(tar_payload) + 1024,
        )
        with pytest.raises(
            utility.AppendAttemptArchiveError,
            match="decompressed size budget",
        ):
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                for _ in archive:
                    pass
            while bounded.read(8 * 1024 * 1024):
                pass


def test_sealed_attempt_builder_produces_validator_compatible_archive(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    builder_work = tmp_path / "builder-work"
    worker_root = builder_work / "worker_outputs"
    worker_root.mkdir(parents=True)
    for name, payload, member_type in synthetic.members:
        if (
            member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}
            and name.startswith("worker_outputs/")
        ):
            relative = name.removeprefix("worker_outputs/")
            destination = worker_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    archive_name = synthetic.archive_path.name
    completed = subprocess.run(
        [
            sys.executable,
            str(ACTIVATION_DIR / "build_attempt_archive.py"),
            "--worker-root",
            "worker_outputs",
            "--job",
            str(synthetic.expected.job_path),
            "--authorization",
            str(synthetic.expected.authorization_path),
            "--activation-manifest",
            str(synthetic.expected.activation_manifest_path),
            "--output-archive",
            archive_name,
            "--execution-id",
            EXECUTION_ID,
            "--cluster-id",
            str(CLUSTER_ID),
            "--proc-id",
            str(PROC_ID),
            "--attempt-ordinal",
            "1",
            "--worker-exit-status",
            "0",
            "--source-archive-sha256",
            utility.SOURCE_ARCHIVE_SHA256,
            "--image-sha256",
            utility.IMAGE_SHA256,
        ],
        cwd=builder_work,
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    builder_result = json.loads(completed.stdout)
    built_archive = builder_work / archive_name
    assert builder_result["status"] == "passed"
    validation = utility.validate_append_attempt_archive(
        built_archive,
        synthetic.expected,
    )
    assert validation["status"] == "passed"
    assert validation["member_validation"]["member_count"] == 12


def test_retrieval_receipt_rejects_remote_local_digest_mismatch(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_append_attempt_archive(
        synthetic.archive_path,
        synthetic.expected,
    )
    remote = replace(
        _remote_observation(synthetic, utility),
        remote_archive_sha256="0" * 64,
    )
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="Remote and local archive hash/size",
    ):
        utility.build_retrieval_receipt(
            validation=validation,
            expected=synthetic.expected,
            submission_receipt_path=SUBMISSION_RECEIPT_PATH,
            remote=remote,
        )


def test_retrieval_receipt_rejects_out_of_order_observations(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_append_attempt_archive(
        synthetic.archive_path,
        synthetic.expected,
    )
    remote = replace(
        _remote_observation(synthetic, utility),
        retrieved_utc="2026-07-31T16:00:00Z",
    )
    with pytest.raises(
        utility.AppendAttemptArchiveError,
        match="timestamps are out of order",
    ):
        utility.build_retrieval_receipt(
            validation=validation,
            expected=synthetic.expected,
            submission_receipt_path=SUBMISSION_RECEIPT_PATH,
            remote=remote,
        )


def test_cli_validates_proc4_and_writes_new_retrieval_receipt(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    remote = _remote_observation(synthetic, utility)
    output = tmp_path / "cli-retrieval.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(UTILITY_PATH),
            "--archive",
            str(synthetic.archive_path),
            "--execution-id",
            EXECUTION_ID,
            "--cluster-id",
            str(CLUSTER_ID),
            "--proc-id",
            str(PROC_ID),
            "--package-manifest",
            str(synthetic.expected.package_manifest_path),
            "--job",
            str(synthetic.expected.job_path),
            "--authorization",
            str(synthetic.expected.authorization_path),
            "--activation-manifest",
            str(synthetic.expected.activation_manifest_path),
            "--receipt-output",
            str(output),
            "--submission-receipt",
            str(SUBMISSION_RECEIPT_PATH),
            "--receipt-created-utc",
            remote.receipt_created_utc,
            "--remote-identity-observed-utc",
            remote.remote_identity_observed_utc,
            "--retrieved-utc",
            remote.retrieved_utc,
            "--remote-archive-sha256",
            remote.remote_archive_sha256,
            "--remote-archive-size-bytes",
            str(remote.remote_archive_size_bytes),
            "--quota-observed-utc",
            remote.quota_observed_utc,
            "--quota-home-used-gib",
            str(remote.quota_home_used_gib),
            "--quota-home-soft-limit-gib",
            str(remote.quota_home_soft_limit_gib),
            "--quota-home-hard-limit-gib",
            str(remote.quota_home_hard_limit_gib),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    receipt = _json(output)
    assert response["status"] == "passed"
    assert response["receipt_output"] == output.as_posix()
    assert response["receipt_sha256"] == receipt["sha256"]
