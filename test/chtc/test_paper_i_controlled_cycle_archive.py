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
    "validate_controlled_cycle_archive.py"
)
EXECUTION_ID = (
    "global_singleton__weak_weak__nph3__"
    "ra_global_singleton_plateau_commutation"
)
CLUSTER_ID = 9397760
PROC_ID = 0
SOURCE_SHA256 = "a" * 64
IMAGE_SHA256 = "b" * 64
CONTROL_PLANE_SHA256 = "c" * 64


def _load_utility() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_controlled_cycle_archive",
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


def _add_regular(
    archive: tarfile.TarFile,
    name: str,
    payload: bytes,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


@dataclass
class SyntheticAttempt:
    archive_path: Path
    job_path: Path
    authorization_path: Path
    activation_path: Path
    submission_path: Path
    members: list[tuple[str, bytes, bytes]]
    expected: Any

    def write_archive(self) -> None:
        with tarfile.open(self.archive_path, "w:gz") as archive:
            for name, payload, member_type in self.members:
                info = tarfile.TarInfo(name)
                info.type = member_type
                if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                else:
                    archive.addfile(info)


def _build_attempt(tmp_path: Path, utility: ModuleType) -> SyntheticAttempt:
    job = _seal(
        utility,
        {
            "schema": "test_job_v1",
            "execution_id": EXECUTION_ID,
            "remote_image": {
                "path": "chtc/phase3_optuna/image.sif",
                "sha256": IMAGE_SHA256,
            },
        },
    )
    job_bytes = _json_bytes(utility, job)
    job_path = tmp_path / "job.json"
    job_path.write_bytes(job_bytes)

    activation_id = "test_controlled_cycle_activation_v1"
    authorization = _seal(
        utility,
        {
            "schema": "test_authorization_v1",
            "activation_id": activation_id,
            "activation_control_plane_sha256": CONTROL_PLANE_SHA256,
            "execution_id": EXECUTION_ID,
            "execution_authorized": True,
            "submission_authorized": True,
            "job_file_sha256": _sha256(job_bytes),
            "job_sha256": job["sha256"],
            "source_archive_sha256": SOURCE_SHA256,
            "remote_image_sha256": IMAGE_SHA256,
        },
    )
    authorization_bytes = _json_bytes(utility, authorization)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_bytes(authorization_bytes)

    activation = _seal(
        utility,
        {
            "schema": "test_activation_v1",
            "activation_id": activation_id,
            "activation_control_plane_sha256": CONTROL_PLANE_SHA256,
            "source_archive_sha256": SOURCE_SHA256,
            "remote_image": {
                "path": "chtc/phase3_optuna/image.sif",
                "sha256": IMAGE_SHA256,
            },
            "executions": [
                {
                    "execution_id": EXECUTION_ID,
                    "job": {
                        "sha256": _sha256(job_bytes),
                        "canonical_sha256": job["sha256"],
                    },
                    "authorization": {
                        "sha256": _sha256(authorization_bytes),
                        "canonical_sha256": authorization["sha256"],
                    },
                }
            ],
        },
    )
    activation_bytes = _json_bytes(utility, activation)
    activation_path = tmp_path / "activation.json"
    activation_path.write_bytes(activation_bytes)

    attempt_ordinal = 1
    worker_files = {
        "attempt_identity.tsv": (
            f"{EXECUTION_ID}\t{CLUSTER_ID}\t{PROC_ID}\t{attempt_ordinal}\n"
        ).encode("utf-8"),
        "result.json": b'{"status":"complete"}\n',
        "runs/test/checkpoints/current.json": b'{"depth":50}\n',
        "worker_exit_status.txt": b"0\n",
    }
    worker_rows = [
        {
            "path": path,
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
        }
        for path, payload in sorted(worker_files.items())
    ]
    attempt_receipt = _seal(
        utility,
        {
            "schema": (
                "paper_i_ra_global_singleton_insertion12_worker_attempt_v1"
            ),
            "execution_id": EXECUTION_ID,
            "cluster_id": CLUSTER_ID,
            "proc_id": PROC_ID,
            "attempt_ordinal": attempt_ordinal,
            "worker_exit_status": 0,
            "job_file_sha256": _sha256(job_bytes),
            "authorization_file_sha256": _sha256(authorization_bytes),
            "activation_manifest_file_sha256": _sha256(activation_bytes),
            "source_archive_sha256": SOURCE_SHA256,
            "image_sha256": IMAGE_SHA256,
            "worker_files": worker_rows,
        },
    )
    attempt_receipt_bytes = _json_bytes(utility, attempt_receipt)
    archive_path = tmp_path / "attempt.tar.gz"
    members = [
        *[
            (f"worker_outputs/{path}", payload, tarfile.REGTYPE)
            for path, payload in sorted(worker_files.items())
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
            attempt_receipt_bytes,
            tarfile.REGTYPE,
        ),
    ]

    submission = _seal(
        utility,
        {
            "schema": "test_submission_receipt_v1",
            "status": "passed",
            "cluster_id": CLUSTER_ID,
            "owner": "jsstrobel",
            "submit_host": "ap2001.chtc.wisc.edu",
            "schedd": "ap2001.chtc.wisc.edu",
            "remote_root": "/home/jsstrobel/Holstein_phase3_optuna_chtc",
            "lifecycle": {
                "mode": "ordinary_held_exact_proc_release_v1",
                "release_scope": "exact_cluster_proc_only",
                "one_proc_per_quota_cycle": True,
            },
            "initial_state": {"proc_ids": list(range(11))},
            "bindings": {
                "activation_manifest": {
                    "sha256": _sha256(activation_bytes),
                    "canonical_sha256": activation["sha256"],
                },
                "source_archive": {"sha256": SOURCE_SHA256},
            },
        },
    )
    submission_path = tmp_path / "submission.json"
    submission_path.write_bytes(_json_bytes(utility, submission))

    synthetic = SyntheticAttempt(
        archive_path=archive_path,
        job_path=job_path,
        authorization_path=authorization_path,
        activation_path=activation_path,
        submission_path=submission_path,
        members=members,
        expected=utility.ExpectedAttempt(
            execution_id=EXECUTION_ID,
            cluster_id=CLUSTER_ID,
            proc_id=PROC_ID,
            job_path=job_path,
            authorization_path=authorization_path,
            activation_manifest_path=activation_path,
            source_archive_sha256=SOURCE_SHA256,
            image_sha256=IMAGE_SHA256,
        ),
    )
    synthetic.write_archive()
    return synthetic


def _remote_observation(
    synthetic: SyntheticAttempt,
    utility: ModuleType,
) -> Any:
    archive_bytes = synthetic.archive_path.read_bytes()
    return utility.RemoteCycleObservation(
        receipt_created_utc="2026-07-31T16:00:00Z",
        retrieved_utc="2026-07-31T15:59:00Z",
        owner="jsstrobel",
        host="ap2001.chtc.wisc.edu",
        schedd="ap2001.chtc.wisc.edu",
        remote_root="/home/jsstrobel/Holstein_phase3_optuna_chtc",
        remote_archive_path=(
            "/home/jsstrobel/Holstein_phase3_optuna_chtc/attempt.tar.gz"
        ),
        remote_archive_sha256=_sha256(archive_bytes),
        remote_archive_size_bytes=len(archive_bytes),
        release_target=f"{CLUSTER_ID}.{PROC_ID}",
        released_utc="2026-07-31T12:00:00Z",
        release_exit_code=0,
        quota_observed_utc="2026-07-31T15:59:30Z",
        quota_home_used_gib=35.03,
        quota_home_soft_limit_gib=40.0,
        quota_home_hard_limit_gib=50.0,
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
    synthetic.write_archive()


def test_valid_archive_streams_without_extraction_and_emits_new_receipt(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_attempt_archive(
        synthetic.archive_path,
        synthetic.expected,
    )

    assert validation["status"] == "passed"
    assert validation["cluster_id"] == CLUSTER_ID
    assert validation["proc_id"] == PROC_ID
    assert validation["member_validation"] == {
        "gzip_and_full_tar_scan_passed": True,
        "safe_unique_regular_only_member_closure_passed": True,
        "worker_inventory_hash_size_closure_passed": True,
        "authority_byte_identity_passed": True,
        "member_count": 8,
        "worker_file_count": 4,
    }
    assert not (tmp_path / "worker_outputs").exists()

    receipt = utility.build_completion_receipt(
        validation=validation,
        expected=synthetic.expected,
        submission_receipt_path=synthetic.submission_path,
        remote=_remote_observation(synthetic, utility),
    )
    utility.verify_self_digest(receipt, label="test receipt")
    assert receipt["schema"] == utility.COMPLETION_RECEIPT_SCHEMA
    assert receipt["release"]["target"] == f"{CLUSTER_ID}.{PROC_ID}"
    assert receipt["quota_after_retrieval"]["soft_limit_headroom_gib"] == 4.97
    output = tmp_path / "completion_receipt_v1.json"
    utility.write_new_receipt(output, receipt)
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(utility.ControlledCycleArchiveError, match="overwrite"):
        utility.write_new_receipt(output, receipt)


@pytest.mark.parametrize(
    "unsafe_kind",
    ["duplicate", "directory", "unsafe_path"],
)
def test_rejects_duplicate_or_nonregular_members(
    tmp_path: Path,
    utility: ModuleType,
    unsafe_kind: str,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    if unsafe_kind == "duplicate":
        synthetic.members.append(
            ("worker_outputs/result.json", b"duplicate", tarfile.REGTYPE)
        )
        match = "Duplicate tar member"
    elif unsafe_kind == "directory":
        synthetic.members.append(("worker_outputs/extra", b"", tarfile.DIRTYPE))
        match = "Non-regular tar member"
    else:
        synthetic.members.append(("../escape.json", b"{}\n", tarfile.REGTYPE))
        match = "Unsafe tar member name"
    synthetic.write_archive()

    with pytest.raises(utility.ControlledCycleArchiveError, match=match):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_worker_inventory_payload_drift(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    _replace_member(
        synthetic,
        "worker_outputs/result.json",
        b'{"status":"tampered"}\n',
    )

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="inventory hash/size mismatch",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_unlisted_worker_file(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    synthetic.members.insert(
        0,
        ("worker_outputs/unlisted.json", b"{}\n", tarfile.REGTYPE),
    )
    synthetic.write_archive()

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="does not close",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_worker_receipt_self_digest_drift(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    receipt_name = "worker_attempt_receipt.json"
    receipt_bytes = next(
        payload for name, payload, _kind in synthetic.members if name == receipt_name
    )
    receipt = json.loads(receipt_bytes)
    receipt["attempt_ordinal"] = 2
    _replace_member(synthetic, receipt_name, _json_bytes(utility, receipt))

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="self digest drifted",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_nonzero_worker_exit_even_with_valid_self_digest(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    receipt_name = "worker_attempt_receipt.json"
    receipt_bytes = next(
        payload for name, payload, _kind in synthetic.members if name == receipt_name
    )
    receipt = json.loads(receipt_bytes)
    receipt.pop("sha256")
    receipt["worker_exit_status"] = 1
    receipt = _seal(utility, receipt)
    _replace_member(synthetic, receipt_name, _json_bytes(utility, receipt))

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="did not exit successfully",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


def test_rejects_exact_proc_identity_mismatch(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    wrong_expected = replace(synthetic.expected, proc_id=1)

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="execution/cluster/proc identity drifted",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            wrong_expected,
        )


def test_rejects_archived_authority_byte_drift(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    _replace_member(synthetic, "authority/job.json", b"{}\n")

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="Archived authority bytes",
    ):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            synthetic.expected,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "source_archive_sha256",
            "d" * 64,
            "authorization relation closure",
        ),
        ("image_sha256", "e" * 64, "Job image binding drifted"),
    ],
)
def test_rejects_source_or_image_binding_drift(
    tmp_path: Path,
    utility: ModuleType,
    field: str,
    value: str,
    match: str,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    wrong_expected = replace(synthetic.expected, **{field: value})

    with pytest.raises(utility.ControlledCycleArchiveError, match=match):
        utility.validate_attempt_archive(
            synthetic.archive_path,
            wrong_expected,
        )


def test_receipt_requires_exact_remote_hash_release_and_submission_identity(
    tmp_path: Path,
    utility: ModuleType,
) -> None:
    synthetic = _build_attempt(tmp_path, utility)
    validation = utility.validate_attempt_archive(
        synthetic.archive_path,
        synthetic.expected,
    )
    remote = _remote_observation(synthetic, utility)

    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="Remote and local archive",
    ):
        utility.build_completion_receipt(
            validation=validation,
            expected=synthetic.expected,
            submission_receipt_path=synthetic.submission_path,
            remote=replace(remote, remote_archive_sha256="d" * 64),
        )
    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="Release target",
    ):
        utility.build_completion_receipt(
            validation=validation,
            expected=synthetic.expected,
            submission_receipt_path=synthetic.submission_path,
            remote=replace(remote, release_target=f"{CLUSTER_ID}.1"),
        )
    with pytest.raises(
        utility.ControlledCycleArchiveError,
        match="remote identity closure",
    ):
        utility.build_completion_receipt(
            validation=validation,
            expected=synthetic.expected,
            submission_receipt_path=synthetic.submission_path,
            remote=replace(remote, owner="another-user"),
        )
