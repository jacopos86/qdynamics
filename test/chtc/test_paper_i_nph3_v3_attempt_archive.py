from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILITY_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "validate_nph3_v3_attempt_archive.py"
)
EXECUTION_ID = (
    "historical_mean_global_singleton_v3_nph3_r50__weak_weak__nph3__"
    "ra_global_singleton_plateau"
)
PROC_ID = 0


def _load_utility() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_nph3_v3_attempt_archive",
        UTILITY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def utility() -> ModuleType:
    return _load_utility()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seal(utility: ModuleType, payload: dict[str, Any]) -> dict[str, Any]:
    return utility.digested(payload)


def _json_bytes(utility: ModuleType, payload: dict[str, Any]) -> bytes:
    return utility.canonical_json_bytes(payload) + b"\n"


@dataclass
class SyntheticAttempt:
    path: Path
    members: list[tuple[str, bytes, bytes]]
    expected: Any

    def write(self) -> None:
        with tarfile.open(self.path, "w:gz") as archive:
            for name, payload, member_type in self.members:
                info = tarfile.TarInfo(name)
                info.type = member_type
                if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    archive.addfile(info)


def _file_binding(payload: bytes) -> dict[str, Any]:
    return {"sha256": _sha256(payload), "size_bytes": len(payload)}


def _build_attempt(
    tmp_path: Path,
    utility: ModuleType,
    *,
    rounds: int = 50,
    manifest_status: str = "passed",
    large_result: bool = False,
) -> SyntheticAttempt:
    expected = utility.ExpectedAttempt(EXECUTION_ID, PROC_ID)
    authority = utility.EXECUTION_AUTHORITIES[EXECUTION_ID]
    job_path, authorization_path, activation_path = utility._authority_paths(
        EXECUTION_ID
    )
    job_bytes = job_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    activation_bytes = activation_path.read_bytes()
    job = json.loads(job_bytes)

    result_payload = (
        b"x" * (utility.SMALL_MEMBER_LIMIT_BYTES + 1)
        if large_result
        else b'{"synthetic":"result"}\n'
    )
    artifacts = {
        "checkpoint.json": b'{"synthetic":"checkpoint"}\n',
        "estimator_ledger.json": b'{"synthetic":"ledger"}\n',
        "paper_i_summary.json": b'{"synthetic":"summary"}\n',
        "result.json": result_payload,
    }
    execution_manifest = _seal(
        utility,
        {
            "schema": utility.EXECUTION_MANIFEST_SCHEMA,
            "status": manifest_status,
            "package_id": utility.PACKAGE_ID,
            "campaign_id": utility.CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": authority.job_canonical_sha256,
            "authorization_sha256": authority.authorization_canonical_sha256,
            "protocol_sha256": job["protocol_sha256"],
            "target_horizon": utility.TARGET_HORIZON,
            "controller_rounds_completed": rounds,
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "output_payloads": {
                name: _file_binding(payload)
                for name, payload in sorted(artifacts.items())
            },
        },
    )
    artifacts["execution_manifest.json"] = _json_bytes(
        utility, execution_manifest
    )
    worker_receipt = _seal(
        utility,
        {
            "schema": utility.WORKER_RECEIPT_SCHEMA,
            "status": "passed",
            "package_id": utility.PACKAGE_ID,
            "campaign_id": utility.CAMPAIGN_ID,
            "execution_id": EXECUTION_ID,
            "job_spec_sha256": authority.job_canonical_sha256,
            "authorization_sha256": authority.authorization_canonical_sha256,
            "execution_manifest_sha256": execution_manifest["sha256"],
            "controller_rounds_completed": rounds,
            "fresh_start": True,
            "artifacts": [
                {"path": name, **_file_binding(payload)}
                for name, payload in sorted(artifacts.items())
            ],
        },
    )
    worker_files = {
        "attempt_identity.tsv": (
            f"{EXECUTION_ID}\t{utility.CLUSTER_ID}\t{PROC_ID}\t1\n"
        ).encode("utf-8"),
        **{f"artifacts/{name}": payload for name, payload in artifacts.items()},
        "worker_exit_status.txt": b"0\n",
        "worker_receipt.json": _json_bytes(utility, worker_receipt),
    }
    attempt_receipt = _seal(
        utility,
        {
            "schema": utility.ATTEMPT_SCHEMA,
            "execution_id": EXECUTION_ID,
            "cluster_id": utility.CLUSTER_ID,
            "proc_id": PROC_ID,
            "attempt_ordinal": 1,
            "worker_exit_status": 0,
            "job_file_sha256": authority.job_file_sha256,
            "authorization_file_sha256": (
                authority.authorization_file_sha256
            ),
            "activation_manifest_file_sha256": (
                utility.ACTIVATION_FILE_SHA256
            ),
            "source_archive_sha256": utility.SOURCE_ARCHIVE_SHA256,
            "image_sha256": utility.IMAGE_SHA256,
            "worker_files": [
                {"path": name, **_file_binding(payload)}
                for name, payload in sorted(worker_files.items())
            ],
        },
    )
    members = [
        *[
            (f"worker_outputs/{name}", payload, tarfile.REGTYPE)
            for name, payload in sorted(worker_files.items())
        ],
        ("authority/job.json", job_bytes, tarfile.REGTYPE),
        (
            "authority/execution_authorization.json",
            authorization_bytes,
            tarfile.REGTYPE,
        ),
        (
            "authority/activation_manifest.json",
            activation_bytes,
            tarfile.REGTYPE,
        ),
        (
            "worker_attempt_receipt.json",
            _json_bytes(utility, attempt_receipt),
            tarfile.REGTYPE,
        ),
    ]
    synthetic = SyntheticAttempt(tmp_path / "attempt.tar.gz", members, expected)
    synthetic.write()
    return synthetic


def _remote_observation(
    synthetic: SyntheticAttempt, utility: ModuleType
) -> Any:
    archive_payload = synthetic.path.read_bytes()
    return utility.RemoteArchiveObservation(
        receipt_created_utc="2026-08-02T21:02:00Z",
        remote_observed_utc="2026-08-02T21:00:00Z",
        retrieved_utc="2026-08-02T21:01:00Z",
        remote_host="ap2001.chtc.wisc.edu",
        remote_archive_path=(
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/transfer/"
            f"{EXECUTION_ID}__{utility.CLUSTER_ID}__{PROC_ID}.tar.gz"
        ),
        remote_archive_sha256=_sha256(archive_payload),
        remote_archive_size_bytes=len(archive_payload),
    )


def _replace_member(
    synthetic: SyntheticAttempt,
    name: str,
    payload: bytes,
) -> None:
    synthetic.members = [
        (member_name, payload if member_name == name else member_payload, kind)
        for member_name, member_payload, kind in synthetic.members
    ]
    synthetic.write()


def test_all_three_sealed_authority_rows_match_exact_local_bytes(
    utility: ModuleType,
) -> None:
    for execution_id, authority in utility.EXECUTION_AUTHORITIES.items():
        expected = utility.ExpectedAttempt(execution_id, authority.proc_id)
        job, authorization, activation = utility._load_authorities(
            expected, authority
        )
        utility._validate_authority_relations(
            expected=expected,
            authority=authority,
            job=job,
            authorization=authorization,
            activation=activation,
        )


def test_streams_large_payload_and_emits_immutable_retrieval_receipt(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility, large_result=True)

    validation = utility.validate_attempt_archive(
        synthetic.path, synthetic.expected
    )

    assert validation["status"] == "passed"
    assert validation["cluster_id"] == utility.CLUSTER_ID
    assert validation["controller_rounds_completed"] == 50
    assert validation["member_validation"][
        "compressed_hash_size_stream_closure_passed"
    ] is True
    assert validation["member_validation"][
        "fifty_round_success_closure_passed"
    ] is True
    assert not (tmp_path / "worker_outputs").exists()

    receipt = utility.build_retrieval_receipt(
        validation=validation,
        expected=synthetic.expected,
        remote=_remote_observation(synthetic, utility),
    )
    utility.verify_self_digest(receipt, label="test retrieval receipt")
    assert receipt["retrieval"]["remote_local_hash_size_match"] is True
    assert receipt["execution"]["controller_rounds_completed"] == 50

    output = tmp_path / "retrieval_receipt.json"
    utility.write_new_receipt(output, receipt)
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(utility.Nph3AttemptArchiveError, match="overwrite"):
        utility.write_new_receipt(output, receipt)


def test_rejects_self_consistent_non_50_round_completion(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility, rounds=49)

    with pytest.raises(
        utility.Nph3AttemptArchiveError,
        match="does not prove 50-round fresh success",
    ):
        utility.validate_attempt_archive(synthetic.path, synthetic.expected)


def test_rejects_failed_execution_manifest_with_closed_hashes(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility, manifest_status="failed")

    with pytest.raises(
        utility.Nph3AttemptArchiveError,
        match="does not prove 50-round fresh success",
    ):
        utility.validate_attempt_archive(synthetic.path, synthetic.expected)


def test_rejects_worker_payload_inventory_drift(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    _replace_member(
        synthetic,
        "worker_outputs/artifacts/result.json",
        b'{"tampered":true}\n',
    )

    with pytest.raises(
        utility.Nph3AttemptArchiveError,
        match="Worker inventory hash/size mismatch",
    ):
        utility.validate_attempt_archive(synthetic.path, synthetic.expected)


def test_rejects_archived_authority_byte_drift(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    _replace_member(synthetic, "authority/job.json", b"{}\n")

    with pytest.raises(
        utility.Nph3AttemptArchiveError,
        match="Archived authority bytes",
    ):
        utility.validate_attempt_archive(synthetic.path, synthetic.expected)


@pytest.mark.parametrize(
    "field", ["remote_archive_sha256", "remote_archive_size_bytes"]
)
def test_retrieval_receipt_rejects_remote_hash_or_size_drift(
    tmp_path: Path,
    utility: ModuleType,
    field: str,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_attempt_archive(
        synthetic.path, synthetic.expected
    )
    remote = _remote_observation(synthetic, utility)
    value: Any = "d" * 64 if field.endswith("sha256") else 1

    with pytest.raises(
        utility.Nph3AttemptArchiveError,
        match="Remote and local archive hash/size",
    ):
        utility.build_retrieval_receipt(
            validation=validation,
            expected=synthetic.expected,
            remote=replace(remote, **{field: value}),
        )
