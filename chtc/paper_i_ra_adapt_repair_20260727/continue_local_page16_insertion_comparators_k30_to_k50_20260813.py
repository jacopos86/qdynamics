#!/usr/bin/env python3
"""Conditionally continue authenticated Page-16 k=30 cells to k=50.

The adapter is local-only.  It never submits work and cannot start a fresh
trajectory.  A cell becomes runnable only after the pinned k=30 runner closes
its result, worker receipt, and effective-plateau gate with the exact decision
``eligible_for_authenticated_resume_to_k50``.  Weak-sector protocols already
have a source-authorized horizon of 50 and are reused byte-for-byte.  A strong-
sector protocol is derived with the sole scientific change of extending the
controller horizon from 30 to 50; its route, insertion policy, problem, source
locks, and all non-horizon request fields remain exact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping, MutableMapping, Sequence


ADAPTER_PATH = Path(__file__).resolve()
REPAIR_ROOT = ADAPTER_PATH.parent
REPO_ROOT = ADAPTER_PATH.parents[2]
K30_RUNNER_PATH = REPAIR_ROOT / (
    "run_local_page16_insertion_comparators_20260812.py"
)
EXPECTED_K30_RUNNER_SHA256 = (
    "bd9d61fb98b48911c3da04faf8b6c38eb391b1a02ab3362e22ef02316a414c4e"
)
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc"
)
K30_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_local_activation"
)
K30_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_20260812_v2"
)
DEFAULT_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_to_k50_"
    "20260813_v2_local_activation"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_to_k50_20260813_v2"
)
RETRIEVED_CHTC_ROOT = REPAIR_ROOT / (
    "retrieved_page16_insertion_comparators_20260812"
)
SW_ALWAYS_CLOSURE_RECEIPT_PATH = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_cluster9647386_sw_always_"
    "remote_materialization_exclusion_receipt_20260813.json"
)
SW_ALWAYS_CLOSURE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_page16_sw_always_"
    "remote_materialization_exclusion_receipt_v2"
)
SW_ALWAYS_CLOSURE_RECEIPT_STATUS = (
    "passed_sw_always_k50_closed_remote_materialization_excluded"
)
SW_ALWAYS_LOCAL_ARCHIVE_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_page16_insertion_comparators_20260812/"
    "strong_weak_u8_always__9647386__1.tar.gz"
)
SW_ALWAYS_REMOTE_ARCHIVE_PATH = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_adapt_page16_insertion_comparators_20260812_v1/outputs/transfer/"
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced__9647386__1.tar.gz"
)

EXPECTED_K30_ACTIVATION_CANONICAL_SHA256 = (
    "e4813a5bcb8e154a621326badc7c922e1e78fccdc3891c8350cb0d36799ebc78"
)
EXPECTED_K30_ACTIVATION_FILE_SHA256 = (
    "7e138a7dcc898f596555bf0839ec987893dc4784c0a7f1d1f01f57504a9f79eb"
)
SOURCE_HORIZON = 30
TARGET_HORIZON = 50
MAX_CONCURRENCY = 1
MIN_FREE_DISK_BYTES = 16 * 1024**3
MIN_MEMORY_PRESSURE_FREE_PERCENT = 20
LOCAL_CHILD_TOKEN_ENV = "PAPER_I_PAGE16_LOCAL_K50_CONTINUATION_SUPERVISOR"
LOCAL_EXECUTION_TARGET = "local_mac_conditional_serial_k30_to_k50_v2"
CONTINUATION_BUNDLE_ID = (
    "paper_i_page16_insertion_comparators_conditional_k30_to_k50_"
    "20260813_v2"
)

ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_request_v2"
)
CONTINUATION_BUNDLE_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_bundle_v2"
)
CONDITIONAL_AUTHORIZATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_conditional_authorization_v2"
)
ACTIVATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_activation_v2"
)
TERMINAL_CHTC_SCHEMA = (
    "paper_i_page16_insertion_comparator_authenticated_chtc_k50_terminal_v2"
)
DECISION_STATUS_SCHEMA = (
    "paper_i_page16_insertion_comparator_k30_decision_status_v2"
)
RESUME_AUTHORIZATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_resume_authorization_v2"
)
RUNTIME_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_runtime_v2"
)
EXECUTION_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_execution_v2"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_worker_receipt_v2"
)
QUARANTINE_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_quarantine_v2"
)
PREFLIGHT_SCHEMA = (
    "paper_i_page16_insertion_comparator_k50_continuation_preflight_v2"
)
TERMINAL_STATUS_SCHEMA = (
    "paper_i_page16_insertion_comparator_authenticated_chtc_terminal_status_v2"
)
SOURCE_WORKER_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_worker_receipt_v1"
)
SOURCE_EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_manifest_v1"
)


class ContinuationError(RuntimeError):
    """A source, authorization, resume, or output contract did not close."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_k30_runner() -> Any:
    if (
        not K30_RUNNER_PATH.is_file()
        or K30_RUNNER_PATH.is_symlink()
        or _sha256_file(K30_RUNNER_PATH) != EXPECTED_K30_RUNNER_SHA256
    ):
        raise ContinuationError("Pinned k30 runner bytes drifted.")
    name = "paper_i_page16_pinned_k30_runner_for_k50_continuation"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, K30_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ContinuationError("Pinned k30 runner cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


k30 = _load_k30_runner()
SW_ALWAYS_CHTC_EXECUTION_ID = k30.WAVES[1][0]
CONDITIONAL_EXECUTION_IDS = tuple(
    execution_id
    for execution_id in k30.TARGET_EXECUTION_IDS
    if execution_id != SW_ALWAYS_CHTC_EXECUTION_ID
)
TERMINAL_CHTC_EXECUTION_IDS = (
    *tuple(k30.COMPLETED_ALWAYS_OPEN_IDS),
    SW_ALWAYS_CHTC_EXECUTION_ID,
)
if (
    len(CONDITIONAL_EXECUTION_IDS) != 9
    or len(TERMINAL_CHTC_EXECUTION_IDS) != 3
    or set(CONDITIONAL_EXECUTION_IDS).intersection(TERMINAL_CHTC_EXECUTION_IDS)
    or set(CONDITIONAL_EXECUTION_IDS).union(TERMINAL_CHTC_EXECUTION_IDS)
    != set(k30.PACKAGE_EXECUTION_IDS)
):
    raise RuntimeError("Hybrid 9-local/3-CHTC campaign inventory drifted.")

TERMINAL_CHTC_ARCHIVES: dict[str, dict[str, Any]] = {
    TERMINAL_CHTC_EXECUTION_IDS[0]: {
        "archive_path": RETRIEVED_CHTC_ROOT
        / "weak_weak_always__9644571__0.tar.gz",
        "archive_sha256": (
            "30ee791a285c7f4413e2f69f9e244053c81354707b55fb39f8054a97a00dc0c0"
        ),
        "archive_size_bytes": 428_656_803,
        "cluster_id": 9_644_571,
        "proc_id": 0,
        "execution_manifest_sha256": (
            "c8d8a797cb88ac3858af3561cea73405d56af7534ab074d1f499e30f45339d27"
        ),
    },
    TERMINAL_CHTC_EXECUTION_IDS[1]: {
        "archive_path": RETRIEVED_CHTC_ROOT
        / "intermediate_weak_always__9647386__0.tar.gz",
        "archive_sha256": (
            "ff20380198dd907b86308832851fe6a450ece26d75c1fccebd60626507066d08"
        ),
        "archive_size_bytes": 395_123_818,
        "cluster_id": 9_647_386,
        "proc_id": 0,
        "execution_manifest_sha256": (
            "05b48ba27801de533454c06fa160cc2d5a97240f915d89b3bdec3de70a805de3"
        ),
    },
}


def _write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(value) + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ContinuationError(f"Stale JSON temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be a JSON object.")
    expected = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if expected != _canonical_sha256(unsigned):
        raise ContinuationError(f"{label} self-digest drifted.")
    return value


def _binding(path: Path, *, root: Path, canonical: bool) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"Cannot bind absent/unsafe file: {path}")
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContinuationError(f"Binding escaped root: {path}") from exc
    row: dict[str, Any] = {
        "path": relative,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        row["canonical_sha256"] = _load_digested(
            path, label=f"binding {relative}"
        )["sha256"]
    return row


def _verify_binding(
    root: Path,
    raw: Any,
    *,
    expected_path: str,
    label: str,
    canonical: bool,
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or raw.get("path") != expected_path:
        raise ContinuationError(f"{label} path binding drifted.")
    path = root / PurePosixPath(expected_path)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or _sha256_file(path) != raw.get("sha256")
    ):
        raise ContinuationError(f"{label} byte binding drifted.")
    if not canonical:
        return None
    value = _load_digested(path, label=label)
    if value.get("sha256") != raw.get("canonical_sha256"):
        raise ContinuationError(f"{label} canonical binding drifted.")
    return value


def _fixed_k30_activation(worker: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, _rows = k30._closed_package(worker)
    activation_path = K30_ACTIVATION_DIR / "activation_manifest.json"
    if (
        not activation_path.is_file()
        or activation_path.is_symlink()
        or _sha256_file(activation_path)
        != EXPECTED_K30_ACTIVATION_FILE_SHA256
    ):
        raise ContinuationError("Pinned k30 activation bytes drifted.")
    activation = k30._validate_activation(
        worker,
        K30_ACTIVATION_DIR,
        manifest=manifest,
    )
    if activation.get("sha256") != EXPECTED_K30_ACTIVATION_CANONICAL_SHA256:
        raise ContinuationError("Pinned k30 activation identity drifted.")
    return manifest, activation


def _job_rows_by_id(worker: Any) -> dict[str, dict[str, Any]]:
    _manifest, rows = k30._closed_package(worker)
    return {str(row["execution_id"]): dict(row) for row in rows}


def _job_by_id(worker: Any) -> dict[str, dict[str, Any]]:
    rows = _job_rows_by_id(worker)
    jobs: dict[str, dict[str, Any]] = {}
    for execution_id in k30.PACKAGE_EXECUTION_IDS:
        job, _manifest, _protocol, _locks = worker._load_closed_job(
            PACKAGE_DIR / str(rows[execution_id]["job_path"])
        )
        jobs[execution_id] = job
    return jobs


def _tar_json_members(
    archive_path: Path,
    *,
    member_names: Sequence[str],
    label: str,
) -> dict[str, dict[str, Any]]:
    expected_names = set(member_names)
    if len(expected_names) != len(member_names):
        raise ContinuationError(f"Duplicate requested {label} member name.")
    observed: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in expected_names:
                continue
            if (
                member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise ContinuationError(f"Unsafe duplicate {label} archive member.")
            stream = archive.extractfile(member)
            if stream is None:
                raise ContinuationError(f"Unreadable {label} archive member.")
            observed[member.name] = stream.read()
    if set(observed) != expected_names:
        raise ContinuationError(f"{label} archive member inventory is incomplete.")
    result: dict[str, dict[str, Any]] = {}
    for name in member_names:
        try:
            value = json.loads(observed[name])
        except json.JSONDecodeError as exc:
            raise ContinuationError(f"{label} is not valid JSON.") from exc
        if not isinstance(value, dict):
            raise ContinuationError(f"{label} must be a JSON object.")
        expected = value.get("sha256")
        unsigned = {key: item for key, item in value.items() if key != "sha256"}
        if expected != _canonical_sha256(unsigned):
            raise ContinuationError(f"{label} self-digest drifted.")
        result[name] = value
    return result


def _authenticate_terminal_chtc_archive(
    worker: Any,
    *,
    execution_id: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    expected = TERMINAL_CHTC_ARCHIVES[execution_id]
    archive_path = Path(expected["archive_path"])
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or archive_path.stat().st_size != expected["archive_size_bytes"]
        or _sha256_file(archive_path) != expected["archive_sha256"]
    ):
        raise ContinuationError(f"Authenticated CHTC archive drifted: {execution_id}")
    member = (
        "./runs/"
        + execution_id
        + "/execution_manifest.json"
    )
    members = _tar_json_members(
        archive_path,
        member_names=(member, "./worker_receipt.json"),
        label=f"CHTC terminal closure {execution_id}",
    )
    manifest = members[member]
    worker_receipt = members["./worker_receipt.json"]
    if (
        manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256")
        != job.get("route_contract_sha256")
        or manifest.get("comparator_policy")
        != "always_commutation_reduced"
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
        or manifest.get("sha256")
        != expected["execution_manifest_sha256"]
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("execution_id") != execution_id
        or worker_receipt.get("job_spec_sha256") != job.get("sha256")
        or worker_receipt.get("execution_manifest_sha256")
        != manifest.get("sha256")
        or worker_receipt.get("controller_rounds_completed")
        != TARGET_HORIZON
        or worker_receipt.get("fresh_start") is not True
    ):
        raise ContinuationError(f"CHTC k50 terminal closure drifted: {execution_id}")
    return _digested(
        {
            "schema": TERMINAL_CHTC_SCHEMA,
            "status": "passed_authenticated_k50_terminal_exclusion",
            "execution_id": execution_id,
            "regime_id": job["regime_id"],
            "nph": int(job["nph"]),
            "comparator_policy": job["comparator_policy"],
            "cluster_id": expected["cluster_id"],
            "proc_id": expected["proc_id"],
            "archive": {
                "path": archive_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": expected["archive_sha256"],
                "size_bytes": expected["archive_size_bytes"],
            },
            "execution_manifest_sha256": manifest["sha256"],
            "worker_receipt_sha256": worker_receipt["sha256"],
            "job_spec_sha256": job["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "controller_rounds_completed": TARGET_HORIZON,
            "continuation_required": False,
            "local_rerun_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _authenticate_sw_always_closure(
    worker: Any,
    *,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_digested(
        SW_ALWAYS_CLOSURE_RECEIPT_PATH,
        label="SW always CHTC closure and remote-materialization-exclusion receipt",
    )
    completed = receipt.get("completed_remote_cell")
    exclusion = receipt.get("remote_materialization_exclusion")
    authentication = receipt.get("authentication")
    if (
        receipt.get("schema") != SW_ALWAYS_CLOSURE_RECEIPT_SCHEMA
        or receipt.get("status") != SW_ALWAYS_CLOSURE_RECEIPT_STATUS
        or not isinstance(completed, Mapping)
        or not isinstance(exclusion, Mapping)
        or not isinstance(authentication, Mapping)
        or receipt.get("scientific_execution_performed_by_action") is not False
    ):
        raise ContinuationError("SW always closure receipt envelope drifted.")

    execution_id = SW_ALWAYS_CHTC_EXECUTION_ID
    archive = completed.get("archive")
    worker_binding = completed.get("worker_receipt")
    manifest_binding = completed.get("execution_manifest")
    history = completed.get("history")
    if (
        completed.get("regime_id") != "strong_weak_u8"
        or completed.get("comparator_policy") != "always_commutation_reduced"
        or completed.get("typed_insertion_kind")
        != job.get("typed_insertion_kind")
        or completed.get("runtime_insertion_mode")
        != job.get("runtime_insertion_mode")
        or completed.get("execution_id") != execution_id
        or type(completed.get("cluster_id")) is not int
        or completed.get("cluster_id") != 9_647_386
        or type(completed.get("proc_id")) is not int
        or completed.get("proc_id") != 1
        or completed.get("controller_rounds_completed") != TARGET_HORIZON
        or completed.get("authenticated_full_sealed_closure") is not True
        or not isinstance(archive, Mapping)
        or not isinstance(worker_binding, Mapping)
        or not isinstance(manifest_binding, Mapping)
        or not isinstance(history, Mapping)
    ):
        raise ContinuationError("SW always completed-cell identity drifted.")

    local_archive_relative = archive.get("path")
    if local_archive_relative != SW_ALWAYS_LOCAL_ARCHIVE_RELATIVE_PATH:
        raise ContinuationError("SW always local archive path drifted.")
    relative = PurePosixPath(str(local_archive_relative))
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ContinuationError("SW always local archive path is unsafe.")
    archive_path = REPO_ROOT.joinpath(*relative.parts)
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ContinuationError("SW always local archive is absent or unsafe.")
    archive_size = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
    size_fields = (
        archive.get("remote_size_bytes"),
        archive.get("local_size_bytes"),
        archive.get("size_bytes"),
    )
    hash_fields = (
        archive.get("remote_sha256"),
        archive.get("local_sha256"),
        archive.get("sha256"),
    )
    if (
        archive.get("remote_path") != SW_ALWAYS_REMOTE_ARCHIVE_PATH
        or any(type(value) is not int for value in size_fields)
        or any(value != archive_size for value in size_fields)
        or any(value != archive_sha256 for value in hash_fields)
    ):
        raise ContinuationError("SW always archive closure binding drifted.")

    expected_manifest_path = f"runs/{execution_id}/execution_manifest.json"
    if (
        worker_binding.get("path_inside_archive") != "worker_receipt.json"
        or worker_binding.get("schema") != SOURCE_WORKER_RECEIPT_SCHEMA
        or worker_binding.get("status") != "passed"
        or manifest_binding.get("path_inside_archive")
        != expected_manifest_path
    ):
        raise ContinuationError("SW always archive member binding drifted.")
    member_names = (
        "./worker_receipt.json",
        f"./{expected_manifest_path}",
    )
    members = _tar_json_members(
        archive_path,
        member_names=member_names,
        label=f"SW always CHTC terminal closure {execution_id}",
    )
    worker_receipt = members[member_names[0]]
    manifest = members[member_names[1]]
    if (
        worker_binding.get("canonical_sha256") != worker_receipt.get("sha256")
        or manifest_binding.get("canonical_sha256") != manifest.get("sha256")
        or worker_receipt.get("schema") != SOURCE_WORKER_RECEIPT_SCHEMA
        or worker_receipt.get("status") != "passed"
        or worker_receipt.get("execution_id") != execution_id
        or worker_receipt.get("job_spec_sha256") != job.get("sha256")
        or worker_receipt.get("execution_manifest_sha256")
        != manifest.get("sha256")
        or worker_receipt.get("controller_rounds_completed") != TARGET_HORIZON
        or worker_receipt.get("fresh_start") is not True
        or manifest.get("schema") != SOURCE_EXECUTION_MANIFEST_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256")
        != job.get("route_contract_sha256")
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
    ):
        raise ContinuationError("SW always sealed worker closure drifted.")

    if (
        type(history.get("cluster_id")) is not int
        or history.get("cluster_id") != 9_647_386
        or type(history.get("proc_id")) is not int
        or history.get("proc_id") != 1
        or type(history.get("job_status")) is not int
        or history.get("job_status") != 4
        or type(history.get("exit_code")) is not int
        or history.get("exit_code") != 0
        or type(history.get("num_job_starts")) is not int
        or history.get("num_job_starts", 0) < 1
        or type(history.get("completion_date_epoch")) is not int
        or history.get("completion_date_epoch", 0) <= 0
    ):
        raise ContinuationError("SW always completed history drifted.")

    before = exclusion.get("before_snapshot")
    after = exclusion.get("after_snapshot")
    outcome = exclusion.get("outcome")
    expected_after: dict[str, Any]
    if outcome == "factory_absent_after_acknowledged_removal":
        expected_after = {
            "cluster_present_in_queue": False,
            "factory_present": False,
            "factory_materialization_paused": None,
            "job_materialize_limit": None,
            "job_materialize_max_idle": None,
            "job_materialize_next_proc_id": None,
            "history_completed_proc_ids": [0, 1],
        }
    elif outcome == (
        "factory_retained_paused_at_completed_prefix_"
        "after_acknowledged_removal"
    ):
        expected_after = {
            "cluster_present_in_queue": False,
            "factory_present": True,
            "factory_materialization_paused": True,
            "job_materialize_limit": 2,
            "job_materialize_max_idle": 0,
            "job_materialize_next_proc_id": 2,
            "history_completed_proc_ids": [0, 1],
        }
    else:
        raise ContinuationError(
            "SW always remote materialization exclusion drifted."
        )
    if (
        exclusion.get("removal_command") != "condor_rm 9647386"
        or exclusion.get("removal_attempts_authenticated") is not True
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or type(before.get("job_materialize_paused")) is not int
        or before.get("job_materialize_paused") != 1
        or type(before.get("job_materialize_next_proc_id")) is not int
        or before.get("job_materialize_next_proc_id") != 2
        or before.get("materialized_proc_ids") != []
        or before.get("history_completed_proc_ids") != [0, 1]
        or dict(after) != expected_after
        or exclusion.get("latent_proc_ids_never_materialized")
        != list(range(2, 11))
        or exclusion.get("queue_cluster_absent") is not True
        or exclusion.get("remote_materialization_excluded") is not True
        or authentication
        != {
            "authenticated_remote_query": True,
            "kind": "interactive_ssh_duo_condor_q_snapshot_v1",
            "source_host": "ap2001.chtc.wisc.edu",
        }
    ):
        raise ContinuationError(
            "SW always remote materialization exclusion drifted."
        )

    return _digested(
        {
            "schema": TERMINAL_CHTC_SCHEMA,
            "status": "passed_authenticated_k50_terminal_exclusion",
            "execution_id": execution_id,
            "regime_id": job["regime_id"],
            "nph": int(job["nph"]),
            "comparator_policy": job["comparator_policy"],
            "cluster_id": 9_647_386,
            "proc_id": 1,
            "archive": {
                "path": local_archive_relative,
                "remote_path": SW_ALWAYS_REMOTE_ARCHIVE_PATH,
                "sha256": archive_sha256,
                "size_bytes": archive_size,
            },
            "execution_manifest_sha256": manifest["sha256"],
            "worker_receipt_sha256": worker_receipt["sha256"],
            "job_spec_sha256": job["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "controller_rounds_completed": TARGET_HORIZON,
            "source_closure_receipt_path": (
                SW_ALWAYS_CLOSURE_RECEIPT_PATH.relative_to(REPO_ROOT).as_posix()
            ),
            "source_closure_receipt_sha256": receipt["sha256"],
            "authenticated_full_sealed_closure": True,
            "remote_materialization_exclusion_outcome": outcome,
            "remote_materialization_exclusion_authenticated": True,
            "continuation_required": False,
            "local_rerun_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _authenticate_terminal_chtc_cell(
    worker: Any,
    *,
    execution_id: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    if execution_id == SW_ALWAYS_CHTC_EXECUTION_ID:
        return _authenticate_sw_always_closure(worker, job=job)
    return _authenticate_terminal_chtc_archive(
        worker,
        execution_id=execution_id,
        job=job,
    )


def terminal_chtc_status(
    *,
    cached: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    worker = k30._load_worker()
    jobs = _job_by_id(worker)
    cache = {} if cached is None else cached
    authenticated: list[dict[str, Any]] = []
    pending: list[str] = []
    errors: dict[str, str] = {}
    sw_receipt_absent = (
        not SW_ALWAYS_CLOSURE_RECEIPT_PATH.exists()
        and not SW_ALWAYS_CLOSURE_RECEIPT_PATH.is_symlink()
    )
    for execution_id in TERMINAL_CHTC_EXECUTION_IDS:
        if execution_id == SW_ALWAYS_CHTC_EXECUTION_ID and sw_receipt_absent:
            pending.append(execution_id)
            continue
        row = cache.get(execution_id)
        if row is None:
            try:
                row = _authenticate_terminal_chtc_cell(
                    worker,
                    execution_id=execution_id,
                    job=jobs[execution_id],
                )
            except (
                ContinuationError,
                OSError,
                ValueError,
                KeyError,
                json.JSONDecodeError,
                tarfile.TarError,
            ) as exc:
                pending.append(execution_id)
                errors[execution_id] = str(exc)
                continue
            cache[execution_id] = row
        authenticated.append(row)
    all_authenticated = (
        len(authenticated) == len(TERMINAL_CHTC_EXECUTION_IDS) and not pending
    )
    if all_authenticated:
        status = "passed_all_three_authenticated_chtc_k50_terminals"
    elif sw_receipt_absent and not errors:
        status = (
            "waiting_for_authenticated_sw_always_closure_and_remote_"
            "materialization_exclusion"
        )
    else:
        status = "blocked_invalid_authenticated_chtc_terminal"
    return _digested(
        {
            "schema": TERMINAL_STATUS_SCHEMA,
            "status": status,
            "all_terminal_cells_authenticated": all_authenticated,
            "terminal_chtc_k50_execution_ids": list(
                TERMINAL_CHTC_EXECUTION_IDS
            ),
            "authenticated_terminal_count": len(authenticated),
            "authenticated_terminal_receipts": authenticated,
            "pending_execution_ids": pending,
            "validation_errors": errors,
            "scientific_execution_performed": False,
        }
    )


def _expected_materialization_requirement(job: Mapping[str, Any]) -> str:
    return (
        "authenticated_resume_adapter_only"
        if int(job["target_horizon"]) >= TARGET_HORIZON
        else "new_source_locked_k50_protocol_required"
    )


def _validate_resume_gate_files(
    worker: Any,
    *,
    job: Mapping[str, Any],
    run_root: Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    decision = gate.get("extension_decision")
    if (
        gate.get("schema") != k30.PLATEAU_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("execution_id") != execution_id
        or gate.get("regime_id") != job.get("regime_id")
        or int(gate.get("nph", -1)) != int(job.get("nph", -2))
        or gate.get("comparator_policy") != job.get("comparator_policy")
        or gate.get("policy") != "paper_i_effective_plateau_v1"
        or gate.get("available_horizon_controller_rounds") != SOURCE_HORIZON
        or decision
        not in {
            "eligible_for_authenticated_resume_to_k50",
            "stop_at_k30",
        }
        or gate.get("source_authorized_horizon")
        != int(job["target_horizon"])
        or gate.get("continuation_target_horizon") != TARGET_HORIZON
        or gate.get("continuation_materialization_requirement")
        != _expected_materialization_requirement(job)
        or gate.get("resume_execution_performed") is not False
        or gate.get("round50_protocol_derived") is not False
    ):
        raise ContinuationError(f"Closed k30 decision gate drifted: {execution_id}")

    checkpoint_row = gate.get("resume_checkpoint")
    sibling_rows = gate.get("resume_checkpoint_siblings")
    if (
        not isinstance(checkpoint_row, Mapping)
        or checkpoint_row.get("path") != "checkpoints/current.json"
        or not isinstance(sibling_rows, list)
        or len(sibling_rows) != 2
    ):
        raise ContinuationError(
            f"Resume checkpoint binding inventory drifted: {execution_id}"
        )
    checkpoint = run_root / "checkpoints/current.json"
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or checkpoint.stat().st_size != int(checkpoint_row.get("size_bytes", -1))
        or worker.sha256_file(checkpoint) != checkpoint_row.get("sha256")
    ):
        raise ContinuationError(f"Resume checkpoint binding drifted: {execution_id}")

    expected_kinds = {
        "estimator_call_ledger_checkpoint": 0,
        "verified_singleton_resume": 0,
    }
    seen_paths: set[str] = set()
    for row in sibling_rows:
        if not isinstance(row, Mapping):
            raise ContinuationError(f"Resume sibling binding malformed: {execution_id}")
        relative = worker.safe_relative_path(
            row.get("path"), label="resume checkpoint sibling"
        )
        name = relative.as_posix()
        path = run_root / relative
        if (
            name in seen_paths
            or relative.parent.as_posix() != "checkpoints"
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or worker.sha256_file(path) != row.get("sha256")
        ):
            raise ContinuationError(f"Resume sibling binding drifted: {execution_id}")
        seen_paths.add(name)
        if relative.name.startswith("current.estimator_call_ledger_checkpoint."):
            expected_kinds["estimator_call_ledger_checkpoint"] += 1
        elif relative.name.startswith("current.verified_singleton_resume."):
            expected_kinds["verified_singleton_resume"] += 1
        else:
            raise ContinuationError(f"Unexpected resume sibling: {execution_id}")
    if set(expected_kinds.values()) != {1}:
        raise ContinuationError(f"Resume sibling role closure drifted: {execution_id}")
    return dict(gate)


def _validated_k30_runtime(worker: Any) -> dict[str, Any] | None:
    _manifest, activation = _fixed_k30_activation(worker)
    if not K30_RUNTIME_DIR.exists() and not K30_RUNTIME_DIR.is_symlink():
        return None
    if K30_RUNTIME_DIR.is_symlink() or not K30_RUNTIME_DIR.is_dir():
        raise ContinuationError("Pinned k30 runtime is unsafe.")
    observed = k30._load_digested(
        worker,
        K30_RUNTIME_DIR / "runtime_manifest.json",
        label="pinned k30 runtime manifest",
    )
    expected = k30._runtime_manifest(worker, activation=activation)
    if observed != expected:
        raise ContinuationError("Pinned k30 runtime manifest drifted.")
    return observed


def _closed_k30_decision(
    worker: Any,
    *,
    execution_id: str,
    job: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not k30._closed_cell(worker, K30_RUNTIME_DIR, execution_id):
        return None
    run_root = K30_RUNTIME_DIR / "runs" / execution_id
    gate_path = K30_RUNTIME_DIR / "plateau_gates" / f"{execution_id}.json"
    receipt_path = K30_RUNTIME_DIR / "worker_receipts" / f"{execution_id}.json"
    manifest_path = run_root / "execution_manifest.json"
    gate = k30._load_digested(worker, gate_path, label=f"k30 gate {execution_id}")
    _validate_resume_gate_files(worker, job=job, run_root=run_root, gate=gate)
    manifest = k30._load_digested(
        worker, manifest_path, label=f"k30 execution manifest {execution_id}"
    )
    receipt = k30._load_digested(
        worker, receipt_path, label=f"k30 worker receipt {execution_id}"
    )
    if (
        manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256")
        != job.get("route_contract_sha256")
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or receipt.get("plateau_gate_sha256") != gate.get("sha256")
    ):
        raise ContinuationError(f"k30 decision provenance drifted: {execution_id}")
    return _digested(
        {
            "schema": "paper_i_page16_k30_authenticated_decision_v2",
            "status": "passed_closed_k30_decision",
            "execution_id": execution_id,
            "extension_decision": gate["extension_decision"],
            "source_authorized_horizon": int(job["target_horizon"]),
            "continuation_materialization_requirement": gate[
                "continuation_materialization_requirement"
            ],
            "k30_runtime_manifest_sha256": _load_digested(
                K30_RUNTIME_DIR / "runtime_manifest.json",
                label="k30 runtime manifest",
            )["sha256"],
            "k30_execution_manifest_sha256": manifest["sha256"],
            "k30_worker_receipt_sha256": receipt["sha256"],
            "k30_plateau_gate_sha256": gate["sha256"],
            "resume_checkpoint": dict(gate["resume_checkpoint"]),
            "resume_checkpoint_siblings": [
                dict(row) for row in gate["resume_checkpoint_siblings"]
            ],
            "run_root": run_root.as_posix(),
            "gate_path": gate_path.as_posix(),
            "receipt_path": receipt_path.as_posix(),
            "manifest_path": manifest_path.as_posix(),
        }
    )


def decision_snapshot(
    *,
    cached: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    worker = k30._load_worker()
    runtime = _validated_k30_runtime(worker)
    jobs = _job_by_id(worker)
    cache = {} if cached is None else cached
    decisions: list[dict[str, Any]] = []
    pending: list[str] = []
    if runtime is None:
        pending.extend(CONDITIONAL_EXECUTION_IDS)
    else:
        for execution_id in CONDITIONAL_EXECUTION_IDS:
            row = cache.get(execution_id)
            if row is None:
                row = _closed_k30_decision(
                    worker,
                    execution_id=execution_id,
                    job=jobs[execution_id],
                )
                if row is not None:
                    cache[execution_id] = row
            if row is None:
                pending.append(execution_id)
            else:
                decisions.append(row)
    eligible = [
        row["execution_id"]
        for row in decisions
        if row["extension_decision"]
        == "eligible_for_authenticated_resume_to_k50"
    ]
    stopped = [
        row["execution_id"]
        for row in decisions
        if row["extension_decision"] == "stop_at_k30"
    ]
    all_closed = len(decisions) == len(CONDITIONAL_EXECUTION_IDS) and not pending
    return _digested(
        {
            "schema": DECISION_STATUS_SCHEMA,
            "status": (
                "passed_all_k30_decisions_closed"
                if all_closed
                else "waiting_for_all_k30_decisions"
            ),
            "all_decisions_closed": all_closed,
            "conditional_execution_ids": list(CONDITIONAL_EXECUTION_IDS),
            "terminal_chtc_k50_execution_ids": list(
                TERMINAL_CHTC_EXECUTION_IDS
            ),
            "closed_decision_count": len(decisions),
            "pending_execution_ids": pending,
            "eligible_execution_ids": eligible,
            "stop_at_k30_execution_ids": stopped,
            "decisions": decisions,
            "scientific_execution_performed": False,
        }
    )


def _non_horizon_protocol_projection(protocol: Any) -> dict[str, Any]:
    payload = copy.deepcopy(protocol.to_dict())
    for key in (
        "sha256",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_materialization",
    ):
        payload.pop(key, None)
    payload.pop("horizon", None)
    payload["request"]["execution"]["stop"].pop(
        "maximum_controller_rounds", None
    )
    payload["stopping_rule"].pop("maximum_controller_rounds", None)
    return payload


def _derive_strong_k50_protocol(
    worker: Any,
    *,
    job: Mapping[str, Any],
    source_protocol: Any,
    continuation_bundle_id: str,
    continuation_bundle_manifest_sha256: str,
) -> Any:
    if (
        int(job["target_horizon"]) != SOURCE_HORIZON
        or int(source_protocol.horizon) != SOURCE_HORIZON
        or source_protocol.request.execution.stop.maximum_controller_rounds
        != SOURCE_HORIZON
        or source_protocol.stopping_rule.get("maximum_controller_rounds")
        != SOURCE_HORIZON
    ):
        raise ContinuationError("Strong source protocol is not exactly horizon 30.")
    payload = copy.deepcopy(source_protocol.to_dict())
    payload["horizon"] = TARGET_HORIZON
    payload["request"]["execution"]["stop"][
        "maximum_controller_rounds"
    ] = TARGET_HORIZON
    payload["stopping_rule"]["maximum_controller_rounds"] = TARGET_HORIZON
    payload["bundle_id"] = continuation_bundle_id
    payload["bundle_manifest_sha256"] = continuation_bundle_manifest_sha256
    receipt = dict(payload["bundle_materialization"])
    receipt.update(
        {
            "bundle_id": continuation_bundle_id,
            "bundle_manifest_sha256": continuation_bundle_manifest_sha256,
            "cell_id": job["execution_id"],
        }
    )
    payload["bundle_materialization"] = worker.digested(receipt)
    payload = worker.digested(payload)

    from pipelines.static_adapt.ra_adapt.contracts import (
        _attach_validated_bundle_protocol_authority,
        _mint_bundle_protocol_materialization_authority,
        resolved_ra_adapt_protocol_from_mapping,
    )

    protocol = resolved_ra_adapt_protocol_from_mapping(payload)
    authority = _mint_bundle_protocol_materialization_authority(
        protocol.bundle_materialization,
        source_lock_refs=protocol.source_locks,
        protocol_sha256=protocol.sha256,
    )
    protocol = _attach_validated_bundle_protocol_authority(protocol, authority)
    if (
        int(protocol.horizon) != TARGET_HORIZON
        or protocol.request.execution.stop.maximum_controller_rounds
        != TARGET_HORIZON
        or protocol.stopping_rule.get("maximum_controller_rounds")
        != TARGET_HORIZON
        or protocol.route_contract != source_protocol.route_contract
        or protocol.request.method != source_protocol.request.method
        or protocol.source_locks != source_protocol.source_locks
        or protocol.algorithm_id != source_protocol.algorithm_id
        or protocol.adapter_id != source_protocol.adapter_id
        or protocol.candidate_representation
        != source_protocol.candidate_representation
        or _non_horizon_protocol_projection(protocol)
        != _non_horizon_protocol_projection(source_protocol)
    ):
        raise ContinuationError(
            "Derived strong protocol changed more than the sole horizon delta."
        )
    return protocol


def prepare_activation(*, activation_dir: Path) -> dict[str, Any]:
    worker = k30._load_worker()
    manifest, k30_activation = _fixed_k30_activation(worker)
    if (
        not SW_ALWAYS_CLOSURE_RECEIPT_PATH.is_file()
        or SW_ALWAYS_CLOSURE_RECEIPT_PATH.is_symlink()
    ):
        raise ContinuationError(
            "Waiting for authenticated SW always closure and remote-"
            "materialization-exclusion receipt."
        )
    if activation_dir.exists() or activation_dir.is_symlink():
        raise FileExistsError(f"Activation destination exists: {activation_dir}")
    activation_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{activation_dir.name}.build-",
            dir=activation_dir.parent,
        )
    )
    try:
        adapter_sha256 = _sha256_file(ADAPTER_PATH)
        jobs = _job_by_id(worker)
        terminal_receipts: dict[str, dict[str, Any]] = {}
        for execution_id in (
            SW_ALWAYS_CHTC_EXECUTION_ID,
            *tuple(k30.COMPLETED_ALWAYS_OPEN_IDS),
        ):
            terminal_receipts[execution_id] = _authenticate_terminal_chtc_cell(
                worker,
                execution_id=execution_id,
                job=jobs[execution_id],
            )
        terminal_bindings: list[dict[str, Any]] = []
        for execution_id in TERMINAL_CHTC_EXECUTION_IDS:
            receipt = terminal_receipts[execution_id]
            path = temporary / "terminal_chtc" / f"{execution_id}.json"
            _write_json(path, receipt, exclusive=True)
            terminal_bindings.append(
                {"execution_id": execution_id, **_binding(path, root=temporary, canonical=True)}
            )

        bundle = _digested(
            {
                "schema": CONTINUATION_BUNDLE_SCHEMA,
                "status": "authorized_conditional_local_continuations",
                "bundle_id": CONTINUATION_BUNDLE_ID,
                "source_package_id": manifest["package_id"],
                "source_package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": k30.SOURCE_ARCHIVE_SHA256,
                "conditional_execution_ids": list(CONDITIONAL_EXECUTION_IDS),
                "terminal_chtc_k50_execution_ids": list(
                    TERMINAL_CHTC_EXECUTION_IDS
                ),
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "maximum_concurrency": MAX_CONCURRENCY,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        bundle_path = temporary / "continuation_bundle_manifest.json"
        _write_json(bundle_path, bundle, exclusive=True)
        bundle_binding = _binding(bundle_path, root=temporary, canonical=True)

        request = _digested(
            {
                "schema": ACTIVATION_REQUEST_SCHEMA,
                "status": "authorized_conditional_local_execution",
                "authorization_kind": "explicit_user_local_execution_authority",
                "explicit_user_authority_recorded": True,
                "authority_date": "2026-08-12",
                "source_package_id": manifest["package_id"],
                "source_package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": k30.SOURCE_ARCHIVE_SHA256,
                "k30_runner_sha256": EXPECTED_K30_RUNNER_SHA256,
                "k30_activation_manifest_sha256": k30_activation["sha256"],
                "local_adapter_sha256": adapter_sha256,
                "conditional_execution_ids": list(CONDITIONAL_EXECUTION_IDS),
                "terminal_chtc_k50_execution_ids": list(
                    TERMINAL_CHTC_EXECUTION_IDS
                ),
                "required_gate_decision": (
                    "eligible_for_authenticated_resume_to_k50"
                ),
                "all_k30_decisions_required_before_first_continuation": True,
                "resume_round": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "strong_protocol_only_scientific_change": {
                    "setting": "maximum_controller_rounds",
                    "before": SOURCE_HORIZON,
                    "after": TARGET_HORIZON,
                },
                "weak_protocol_reused_without_scientific_change": True,
                "accepted_state_resume_required": True,
                "fresh_start_authorized": False,
                "resume_ledger_sidecar_required": True,
                "verified_resume_sidecar_required": True,
                "maximum_concurrency": MAX_CONCURRENCY,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        request_path = temporary / "activation_request.json"
        _write_json(request_path, request, exclusive=True)
        request_binding = _binding(request_path, root=temporary, canonical=True)

        authorization_bindings: list[dict[str, Any]] = []
        rows = _job_rows_by_id(worker)
        for execution_id in CONDITIONAL_EXECUTION_IDS:
            job, _package, source_protocol, _problem, prepared = worker._prepare(
                PACKAGE_DIR / rows[execution_id]["job_path"]
            )
            try:
                target_protocol: dict[str, Any]
                if int(job["target_horizon"]) == TARGET_HORIZON:
                    target_protocol = {
                        "kind": "source_protocol_reused_exactly",
                        "source_protocol_sha256": source_protocol.sha256,
                        "target_protocol_sha256": source_protocol.sha256,
                        "path": rows[execution_id]["protocol_path"],
                    }
                else:
                    protocol = _derive_strong_k50_protocol(
                        worker,
                        job=job,
                        source_protocol=source_protocol,
                        continuation_bundle_id=CONTINUATION_BUNDLE_ID,
                        continuation_bundle_manifest_sha256=bundle["sha256"],
                    )
                    protocol_path = temporary / "protocols" / f"{execution_id}.json"
                    _write_json(
                        protocol_path,
                        protocol.to_dict(),
                        exclusive=True,
                    )
                    target_protocol = {
                        "kind": "source_locked_sole_horizon_delta_30_to_50",
                        "source_protocol_sha256": source_protocol.sha256,
                        "target_protocol_sha256": protocol.sha256,
                        **_binding(
                            protocol_path,
                            root=temporary,
                            canonical=True,
                        ),
                    }
            finally:
                prepared.cleanup()
            authority = _digested(
                {
                    "schema": CONDITIONAL_AUTHORIZATION_SCHEMA,
                    "status": "authorized_only_if_closed_gate_is_eligible",
                    "execution_id": execution_id,
                    "regime_id": job["regime_id"],
                    "nph": int(job["nph"]),
                    "comparator_policy": job["comparator_policy"],
                    "job_spec_sha256": job["sha256"],
                    "source_protocol_sha256": source_protocol.sha256,
                    "target_protocol": target_protocol,
                    "route_contract_sha256": job["route_contract_sha256"],
                    "source_authorized_horizon": int(job["target_horizon"]),
                    "resume_round": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "required_gate_schema": k30.PLATEAU_GATE_SCHEMA,
                    "required_gate_decision": (
                        "eligible_for_authenticated_resume_to_k50"
                    ),
                    "continuation_materialization_requirement": (
                        _expected_materialization_requirement(job)
                    ),
                    "accepted_state_resume_required": True,
                    "fresh_start_authorized": False,
                    "maximum_concurrency": MAX_CONCURRENCY,
                    "local_adapter_sha256": adapter_sha256,
                    "k30_runner_sha256": EXPECTED_K30_RUNNER_SHA256,
                    "continuation_bundle_manifest_sha256": bundle["sha256"],
                    "execution_authorized": True,
                    "submission_authorized": False,
                    "paper_evidence_adoption_authorized": False,
                }
            )
            path = temporary / "conditional_authorizations" / f"{execution_id}.json"
            _write_json(path, authority, exclusive=True)
            authorization_bindings.append(
                {"execution_id": execution_id, **_binding(path, root=temporary, canonical=True)}
            )

        activation = _digested(
            {
                "schema": ACTIVATION_SCHEMA,
                "status": "passed_conditional_activation_prepared_no_execution",
                "source_package_id": manifest["package_id"],
                "source_package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": k30.SOURCE_ARCHIVE_SHA256,
                "k30_runner_sha256": EXPECTED_K30_RUNNER_SHA256,
                "k30_activation_manifest_sha256": k30_activation["sha256"],
                "local_adapter_sha256": adapter_sha256,
                "activation_request": request_binding,
                "continuation_bundle_manifest": bundle_binding,
                "conditional_authorizations": authorization_bindings,
                "authorization_count": len(authorization_bindings),
                "terminal_chtc_receipts": terminal_bindings,
                "terminal_chtc_count": len(terminal_bindings),
                "conditional_execution_ids": list(CONDITIONAL_EXECUTION_IDS),
                "terminal_chtc_k50_execution_ids": list(
                    TERMINAL_CHTC_EXECUTION_IDS
                ),
                "all_k30_decisions_required_before_first_continuation": True,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "maximum_concurrency": MAX_CONCURRENCY,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "scientific_execution_performed": False,
                "submitted": False,
            }
        )
        _write_json(
            temporary / "activation_manifest.json",
            activation,
            exclusive=True,
        )
        os.rename(temporary, activation_dir)
        return activation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_activation(
    worker: Any,
    activation_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, k30_activation = _fixed_k30_activation(worker)
    if activation_dir.is_symlink() or not activation_dir.is_dir():
        raise ContinuationError("Continuation activation is absent or unsafe.")
    activation = _load_digested(
        activation_dir / "activation_manifest.json",
        label="continuation activation manifest",
    )
    request = _verify_binding(
        activation_dir,
        activation.get("activation_request"),
        expected_path="activation_request.json",
        label="continuation activation request",
        canonical=True,
    )
    bundle = _verify_binding(
        activation_dir,
        activation.get("continuation_bundle_manifest"),
        expected_path="continuation_bundle_manifest.json",
        label="continuation bundle manifest",
        canonical=True,
    )
    assert request is not None and bundle is not None
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status")
        != "passed_conditional_activation_prepared_no_execution"
        or activation.get("source_package_manifest_sha256")
        != manifest.get("sha256")
        or activation.get("source_archive_sha256")
        != k30.SOURCE_ARCHIVE_SHA256
        or activation.get("k30_runner_sha256")
        != EXPECTED_K30_RUNNER_SHA256
        or activation.get("k30_activation_manifest_sha256")
        != k30_activation.get("sha256")
        or activation.get("local_adapter_sha256") != _sha256_file(ADAPTER_PATH)
        or activation.get("conditional_execution_ids")
        != list(CONDITIONAL_EXECUTION_IDS)
        or activation.get("terminal_chtc_k50_execution_ids")
        != list(TERMINAL_CHTC_EXECUTION_IDS)
        or activation.get("authorization_count")
        != len(CONDITIONAL_EXECUTION_IDS)
        or activation.get("terminal_chtc_count")
        != len(TERMINAL_CHTC_EXECUTION_IDS)
        or activation.get("source_horizon") != SOURCE_HORIZON
        or activation.get("target_horizon") != TARGET_HORIZON
        or activation.get("maximum_concurrency") != MAX_CONCURRENCY
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("submitted") is not False
        or request.get("schema") != ACTIVATION_REQUEST_SCHEMA
        or request.get("required_gate_decision")
        != "eligible_for_authenticated_resume_to_k50"
        or request.get("all_k30_decisions_required_before_first_continuation")
        is not True
        or request.get("accepted_state_resume_required") is not True
        or request.get("fresh_start_authorized") is not False
        or bundle.get("schema") != CONTINUATION_BUNDLE_SCHEMA
        or bundle.get("bundle_id") != CONTINUATION_BUNDLE_ID
        or bundle.get("maximum_concurrency") != MAX_CONCURRENCY
    ):
        raise ContinuationError("Continuation activation contract drifted.")

    auth_rows = activation.get("conditional_authorizations")
    terminal_rows = activation.get("terminal_chtc_receipts")
    if (
        not isinstance(auth_rows, list)
        or [row.get("execution_id") for row in auth_rows]
        != list(CONDITIONAL_EXECUTION_IDS)
        or not isinstance(terminal_rows, list)
        or [row.get("execution_id") for row in terminal_rows]
        != list(TERMINAL_CHTC_EXECUTION_IDS)
    ):
        raise ContinuationError("Continuation activation inventory drifted.")
    for row in auth_rows:
        execution_id = str(row["execution_id"])
        authority = _verify_binding(
            activation_dir,
            row,
            expected_path=f"conditional_authorizations/{execution_id}.json",
            label=f"conditional authorization {execution_id}",
            canonical=True,
        )
        assert authority is not None
        if (
            authority.get("schema") != CONDITIONAL_AUTHORIZATION_SCHEMA
            or authority.get("execution_id") != execution_id
            or authority.get("required_gate_decision")
            != "eligible_for_authenticated_resume_to_k50"
            or authority.get("fresh_start_authorized") is not False
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not False
            or authority.get("local_adapter_sha256")
            != activation.get("local_adapter_sha256")
            or authority.get("continuation_bundle_manifest_sha256")
            != bundle.get("sha256")
        ):
            raise ContinuationError(
                f"Conditional authorization drifted: {execution_id}"
            )
        target_protocol = authority.get("target_protocol")
        if not isinstance(target_protocol, Mapping):
            raise ContinuationError(
                f"Conditional target protocol is absent: {execution_id}"
            )
        if target_protocol.get("kind") == "source_protocol_reused_exactly":
            if (
                target_protocol.get("source_protocol_sha256")
                != authority.get("source_protocol_sha256")
                or target_protocol.get("target_protocol_sha256")
                != authority.get("source_protocol_sha256")
            ):
                raise ContinuationError(
                    f"Weak target protocol reuse drifted: {execution_id}"
                )
        elif target_protocol.get("kind") == (
            "source_locked_sole_horizon_delta_30_to_50"
        ):
            protocol_path = f"protocols/{execution_id}.json"
            protocol_payload = _verify_binding(
                activation_dir,
                target_protocol,
                expected_path=protocol_path,
                label=f"strong target protocol {execution_id}",
                canonical=True,
            )
            assert protocol_payload is not None
            if (
                protocol_payload.get("sha256")
                != target_protocol.get("target_protocol_sha256")
                or protocol_payload.get("horizon") != TARGET_HORIZON
            ):
                raise ContinuationError(
                    f"Strong target protocol drifted: {execution_id}"
                )
        else:
            raise ContinuationError(
                f"Unknown target protocol materialization: {execution_id}"
            )
    for row in terminal_rows:
        execution_id = str(row["execution_id"])
        receipt = _verify_binding(
            activation_dir,
            row,
            expected_path=f"terminal_chtc/{execution_id}.json",
            label=f"terminal CHTC receipt {execution_id}",
            canonical=True,
        )
        assert receipt is not None
        if (
            receipt.get("schema") != TERMINAL_CHTC_SCHEMA
            or receipt.get("execution_id") != execution_id
            or receipt.get("controller_rounds_completed") != TARGET_HORIZON
            or receipt.get("continuation_required") is not False
            or receipt.get("local_rerun_authorized") is not False
        ):
            raise ContinuationError(f"Terminal CHTC receipt drifted: {execution_id}")
        if execution_id == SW_ALWAYS_CHTC_EXECUTION_ID and (
            receipt.get("source_closure_receipt_sha256") is None
            or receipt.get("authenticated_full_sealed_closure") is not True
            or receipt.get("remote_materialization_exclusion_authenticated")
            is not True
        ):
            raise ContinuationError(
                "SW always terminal receipt lacks authenticated remote "
                "materialization exclusion."
            )
    return activation, bundle


def _free_disk_bytes(path: Path) -> int:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ContinuationError(f"No existing parent for disk check: {path}")
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free


def capacity_receipt(*, runtime_dir: Path) -> dict[str, Any]:
    available_memory = k30._available_memory_bytes()
    pressure_free = k30._memory_pressure_free_percent()
    free_disk = _free_disk_bytes(runtime_dir.parent)
    blockers: list[str] = []
    if available_memory is None:
        blockers.append("available_memory_unavailable")
    elif available_memory <= 0:
        blockers.append("available_memory_nonpositive")
    if pressure_free is None:
        blockers.append("memory_pressure_unavailable")
    elif pressure_free < MIN_MEMORY_PRESSURE_FREE_PERCENT:
        blockers.append("memory_pressure_free_percentage_below_guard")
    if free_disk < MIN_FREE_DISK_BYTES:
        blockers.append("free_disk_below_guard")
    return _digested(
        {
            "schema": (
                "paper_i_page16_insertion_comparator_k50_"
                "continuation_capacity_v2"
            ),
            "status": "passed" if not blockers else "blocked",
            "physical_memory_bytes": k30._physical_memory_bytes(),
            "available_or_reclaimable_memory_bytes": available_memory,
            "memory_pressure_free_percent": pressure_free,
            "swap_usage": k30._swap_usage(),
            "free_disk_bytes": free_disk,
            "required_memory_pressure_free_percent": (
                MIN_MEMORY_PRESSURE_FREE_PERCENT
            ),
            "required_free_disk_bytes": MIN_FREE_DISK_BYTES,
            "maximum_concurrency": MAX_CONCURRENCY,
            "blockers": blockers,
            "scientific_execution_performed": False,
        }
    )


def _overlapping_scientific_commands() -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command=", "-ww"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContinuationError("Cannot audit local scientific overlap.") from exc
    own_pid = os.getpid()
    matches: list[str] = []
    for raw in output.splitlines():
        text = raw.strip()
        pid_text, _, command = text.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        if (
            ADAPTER_PATH.as_posix() in command
            and "--run-cell" in command
        ):
            matches.append(text)
    return matches


def inert_preflight(
    *,
    activation_dir: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    worker = k30._load_worker()
    manifest, _k30_activation = _fixed_k30_activation(worker)
    activation_status = "absent"
    if activation_dir.exists() or activation_dir.is_symlink():
        _validate_activation(worker, activation_dir)
        activation_status = "validated"
    snapshot = decision_snapshot()
    terminal_status = terminal_chtc_status()
    capacity = capacity_receipt(runtime_dir=runtime_dir)
    overlap = _overlapping_scientific_commands()
    runtime_collision = runtime_dir.exists() or runtime_dir.is_symlink()
    blockers: list[str] = []
    blockers.extend(capacity["blockers"])
    if activation_status != "validated":
        blockers.append("continuation_activation_absent")
    if not snapshot["all_decisions_closed"]:
        blockers.append("local_k30_decisions_pending")
    if not terminal_status["all_terminal_cells_authenticated"]:
        blockers.append(terminal_status["status"])
    if overlap:
        blockers.append("overlapping_local_continuation_worker")
    if runtime_collision:
        blockers.append("continuation_runtime_already_exists")
    return _digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": (
                "passed_ready_for_supervised_initialization"
                if activation_status == "validated"
                and snapshot["all_decisions_closed"]
                and terminal_status["all_terminal_cells_authenticated"]
                and not blockers
                else "passed_waiting_or_inert"
            ),
            "source_package_id": manifest["package_id"],
            "source_package_manifest_sha256": manifest["sha256"],
            "k30_runner_sha256": EXPECTED_K30_RUNNER_SHA256,
            "local_adapter_sha256": _sha256_file(ADAPTER_PATH),
            "activation_status": activation_status,
            "decision_status": snapshot,
            "terminal_chtc_status": terminal_status,
            "terminal_chtc_k50_execution_ids": list(
                TERMINAL_CHTC_EXECUTION_IDS
            ),
            "capacity": capacity,
            "overlapping_scientific_commands": overlap,
            "runtime_collision": runtime_collision,
            "blockers": blockers,
            "run_ready": (
                activation_status == "validated"
                and snapshot["all_decisions_closed"]
                and terminal_status["all_terminal_cells_authenticated"]
                and not blockers
            ),
            "maximum_concurrency": MAX_CONCURRENCY,
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def _conditional_authorization(
    activation_dir: Path,
    activation: Mapping[str, Any],
    execution_id: str,
) -> dict[str, Any]:
    row = next(
        row
        for row in activation["conditional_authorizations"]
        if row["execution_id"] == execution_id
    )
    authority = _verify_binding(
        activation_dir,
        row,
        expected_path=f"conditional_authorizations/{execution_id}.json",
        label=f"conditional authorization {execution_id}",
        canonical=True,
    )
    assert authority is not None
    return authority


def initialize_runtime(
    *,
    activation_dir: Path,
    runtime_dir: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    worker = k30._load_worker()
    activation, bundle = _validate_activation(worker, activation_dir)
    if snapshot.get("all_decisions_closed") is not True:
        raise ContinuationError(
            "All nine k30 decisions must close before runtime initialization."
        )
    decisions = snapshot.get("decisions")
    if (
        not isinstance(decisions, list)
        or [row.get("execution_id") for row in decisions]
        != list(CONDITIONAL_EXECUTION_IDS)
    ):
        raise ContinuationError("Closed k30 decision order or identity drifted.")
    if runtime_dir.exists() or runtime_dir.is_symlink():
        raise FileExistsError(f"Continuation runtime exists: {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{runtime_dir.name}.build-",
            dir=runtime_dir.parent,
        )
    )
    try:
        for name in (
            "runs",
            "worker_receipts",
            "authorizations",
            "logs",
            "status",
            "in_progress",
            "quarantine",
        ):
            (temporary / name).mkdir()
        jobs = _job_by_id(worker)
        decision_by_id = {row["execution_id"]: row for row in decisions}
        eligible = list(snapshot["eligible_execution_ids"])
        stopped = list(snapshot["stop_at_k30_execution_ids"])
        authorization_bindings: list[dict[str, Any]] = []
        for execution_id in eligible:
            decision = decision_by_id[execution_id]
            job = jobs[execution_id]
            conditional = _conditional_authorization(
                activation_dir, activation, execution_id
            )
            if (
                decision.get("extension_decision")
                != "eligible_for_authenticated_resume_to_k50"
                or conditional.get("job_spec_sha256") != job.get("sha256")
                or conditional.get("route_contract_sha256")
                != job.get("route_contract_sha256")
            ):
                raise ContinuationError(f"Eligible decision drifted: {execution_id}")
            authority = _digested(
                {
                    "schema": RESUME_AUTHORIZATION_SCHEMA,
                    "status": "authorized_authenticated_resume_to_k50",
                    "execution_id": execution_id,
                    "job_spec_sha256": job["sha256"],
                    "source_protocol_sha256": job["protocol_sha256"],
                    "target_protocol": conditional["target_protocol"],
                    "route_contract_sha256": job["route_contract_sha256"],
                    "comparator_policy": job["comparator_policy"],
                    "regime_id": job["regime_id"],
                    "nph": int(job["nph"]),
                    "source_authorized_horizon": int(job["target_horizon"]),
                    "resume_round": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "continuation_materialization_requirement": decision[
                        "continuation_materialization_requirement"
                    ],
                    "k30_execution_manifest_sha256": decision[
                        "k30_execution_manifest_sha256"
                    ],
                    "k30_worker_receipt_sha256": decision[
                        "k30_worker_receipt_sha256"
                    ],
                    "k30_plateau_gate_sha256": decision[
                        "k30_plateau_gate_sha256"
                    ],
                    "resume_checkpoint": decision["resume_checkpoint"],
                    "resume_checkpoint_siblings": decision[
                        "resume_checkpoint_siblings"
                    ],
                    "source_run_root": decision["run_root"],
                    "conditional_authorization_sha256": conditional["sha256"],
                    "activation_manifest_sha256": activation["sha256"],
                    "continuation_bundle_manifest_sha256": bundle["sha256"],
                    "accepted_state_resume_required": True,
                    "fresh_start_authorized": False,
                    "execution_authorized": True,
                    "submission_authorized": False,
                    "paper_evidence_adoption_authorized": False,
                }
            )
            path = temporary / "authorizations" / f"{execution_id}.json"
            _write_json(path, authority, exclusive=True)
            authorization_bindings.append(
                {"execution_id": execution_id, **_binding(path, root=temporary, canonical=True)}
            )
        runtime = _digested(
            {
                "schema": RUNTIME_SCHEMA,
                "status": "authorized_pending_conditional_continuations",
                "adapter_path": ADAPTER_PATH.relative_to(REPO_ROOT).as_posix(),
                "adapter_sha256": _sha256_file(ADAPTER_PATH),
                "activation_manifest_sha256": activation["sha256"],
                "continuation_bundle_manifest_sha256": bundle["sha256"],
                "k30_runner_sha256": EXPECTED_K30_RUNNER_SHA256,
                "k30_runtime_manifest_sha256": decisions[0][
                    "k30_runtime_manifest_sha256"
                ],
                "decision_status_sha256": snapshot["sha256"],
                "conditional_execution_ids": list(CONDITIONAL_EXECUTION_IDS),
                "eligible_execution_ids": eligible,
                "stop_at_k30_execution_ids": stopped,
                "terminal_chtc_k50_execution_ids": list(
                    TERMINAL_CHTC_EXECUTION_IDS
                ),
                "resume_authorizations": authorization_bindings,
                "authorization_count": len(authorization_bindings),
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "maximum_concurrency": MAX_CONCURRENCY,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json(temporary / "runtime_manifest.json", runtime, exclusive=True)
        _write_json(
            temporary / "status/decision_status.json",
            snapshot,
            exclusive=True,
        )
        os.rename(temporary, runtime_dir)
        return runtime
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_runtime(
    worker: Any,
    *,
    activation_dir: Path,
    runtime_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    activation, bundle = _validate_activation(worker, activation_dir)
    if runtime_dir.is_symlink() or not runtime_dir.is_dir():
        raise ContinuationError("Continuation runtime is absent or unsafe.")
    runtime = _load_digested(
        runtime_dir / "runtime_manifest.json",
        label="continuation runtime manifest",
    )
    if (
        runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("adapter_sha256") != _sha256_file(ADAPTER_PATH)
        or runtime.get("activation_manifest_sha256") != activation.get("sha256")
        or runtime.get("continuation_bundle_manifest_sha256")
        != bundle.get("sha256")
        or runtime.get("conditional_execution_ids")
        != list(CONDITIONAL_EXECUTION_IDS)
        or runtime.get("terminal_chtc_k50_execution_ids")
        != list(TERMINAL_CHTC_EXECUTION_IDS)
        or runtime.get("maximum_concurrency") != MAX_CONCURRENCY
        or runtime.get("source_horizon") != SOURCE_HORIZON
        or runtime.get("target_horizon") != TARGET_HORIZON
        or runtime.get("execution_authorized") is not True
        or runtime.get("submission_authorized") is not False
    ):
        raise ContinuationError("Continuation runtime manifest drifted.")
    eligible = runtime.get("eligible_execution_ids")
    stopped = runtime.get("stop_at_k30_execution_ids")
    rows = runtime.get("resume_authorizations")
    if (
        not isinstance(eligible, list)
        or not isinstance(stopped, list)
        or set(eligible).intersection(stopped)
        or set(eligible).union(stopped) != set(CONDITIONAL_EXECUTION_IDS)
        or not isinstance(rows, list)
        or [row.get("execution_id") for row in rows] != eligible
        or runtime.get("authorization_count") != len(eligible)
    ):
        raise ContinuationError("Continuation runtime decision inventory drifted.")
    return runtime, activation, bundle


def _resume_authorization(
    worker: Any,
    *,
    activation_dir: Path,
    runtime_dir: Path,
    execution_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime, activation, bundle = _validate_runtime(
        worker,
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
    )
    if execution_id not in runtime["eligible_execution_ids"]:
        raise ContinuationError(f"Cell is not eligible for k50: {execution_id}")
    row = next(
        row
        for row in runtime["resume_authorizations"]
        if row["execution_id"] == execution_id
    )
    authority = _verify_binding(
        runtime_dir,
        row,
        expected_path=f"authorizations/{execution_id}.json",
        label=f"resume authorization {execution_id}",
        canonical=True,
    )
    assert authority is not None
    if (
        authority.get("schema") != RESUME_AUTHORIZATION_SCHEMA
        or authority.get("status") != "authorized_authenticated_resume_to_k50"
        or authority.get("execution_id") != execution_id
        or authority.get("activation_manifest_sha256") != activation.get("sha256")
        or authority.get("continuation_bundle_manifest_sha256")
        != bundle.get("sha256")
        or authority.get("accepted_state_resume_required") is not True
        or authority.get("fresh_start_authorized") is not False
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not False
    ):
        raise ContinuationError(f"Resume authorization drifted: {execution_id}")
    return authority, runtime, bundle


def _execute_resume(
    *,
    protocol: Any,
    problem: Any,
    checkpoint: Path,
    checkpoint_sha256: str,
    staging: Path,
) -> tuple[Any, int]:
    from pipelines.static_adapt.ra_adapt import (
        RAAdaptOperationalControls,
        run_ra_adapt,
    )
    from pipelines.static_adapt.ra_adapt import engine as ra_engine
    from pipelines.reporting import paper_i_run_summary as summary_module
    from pipelines.static_adapt.sr_snake import (
        AcceptedStateResume,
        CheckpointObservation,
        EstimatorLedgerObservation,
        SRObservationPolicy,
    )

    (staging / "checkpoints").mkdir(parents=True, exist_ok=False)
    (staging / "result").mkdir(parents=True, exist_ok=False)
    (staging / "summary").mkdir(parents=True, exist_ok=False)
    (staging / "continuation").mkdir(parents=True, exist_ok=False)
    controls = RAAdaptOperationalControls(
        maximum_controller_rounds=TARGET_HORIZON,
        resume=AcceptedStateResume(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        ),
        observation=SRObservationPolicy(
            checkpoint=CheckpointObservation(
                path=staging / "checkpoints/current.json",
                every_controller_rounds=1,
                keep_history_tail=100,
            ),
            estimator_ledger=EstimatorLedgerObservation(
                path=staging / "result/estimator_ledger.json"
            ),
            resource_rounds=(TARGET_HORIZON,),
        ),
    )
    bound_route = dict(protocol.route_contract)
    bound_route_sha256 = str(bound_route.pop("sha256"))
    bound_route_profile = str(bound_route["route_profile"])
    original_route_builder = ra_engine._repaired_route_contract
    original_reduction_validator = (
        ra_engine.validate_commutation_reduced_insertion_receipt
    )
    original_summary_identities = (
        summary_module._canonical_ra_supersession_identities
    )

    def comparator_route_builder(
        request: Any,
        *,
        active_gradient_policy: str,
        resource_weighting_scope: str,
        algorithm_id: str | None = None,
        problem: Any = None,
    ) -> tuple[str, str, dict[str, Any], str]:
        if (
            str(algorithm_id) == str(protocol.algorithm_id)
            and request.method.insertion.kind
            in {"always_commutation_reduced", "append_only"}
        ):
            return (
                bound_route_profile,
                bound_route_profile,
                dict(bound_route),
                bound_route_sha256,
            )
        return original_route_builder(
            request,
            active_gradient_policy=active_gradient_policy,
            resource_weighting_scope=resource_weighting_scope,
            algorithm_id=algorithm_id,
            problem=problem,
        )

    def comparator_summary_identities(
        method: Any,
        *,
        candidate_representation: str,
    ) -> tuple[tuple[str, str, str, str], ...]:
        if (
            method.insertion.kind
            in {"always_commutation_reduced", "append_only"}
            and candidate_representation == protocol.candidate_representation
        ):
            return ((
                "ra_adapt",
                bound_route_profile,
                bound_route_profile,
                bound_route_sha256,
            ),)
        return original_summary_identities(
            method,
            candidate_representation=candidate_representation,
        )

    def comparator_reduction_validator(
        receipt: Mapping[str, Any],
        *,
        expected_policy: str,
        expected_requested_positions: Any = None,
        scored_population: Any = None,
    ) -> dict[str, Any]:
        return original_reduction_validator(
            receipt,
            expected_policy=expected_policy,
            expected_requested_positions=expected_requested_positions,
            scored_population=(
                None
                if protocol.request.method.insertion.kind
                == "always_commutation_reduced"
                else scored_population
            ),
        )

    ra_engine._repaired_route_contract = comparator_route_builder
    ra_engine.validate_commutation_reduced_insertion_receipt = (
        comparator_reduction_validator
    )
    summary_module._canonical_ra_supersession_identities = (
        comparator_summary_identities
    )
    try:
        result = run_ra_adapt(
            problem,
            protocol,
            operational_controls=controls,
        )
    finally:
        ra_engine._repaired_route_contract = original_route_builder
        ra_engine.validate_commutation_reduced_insertion_receipt = (
            original_reduction_validator
        )
        summary_module._canonical_ra_supersession_identities = (
            original_summary_identities
        )
    rounds = len(result.run.accepted_trajectory)
    if (
        result.protocol.sha256 != protocol.sha256
        or rounds != TARGET_HORIZON
        or not (staging / "checkpoints/current.json").is_file()
        or not (staging / "result/estimator_ledger.json").is_file()
    ):
        raise ContinuationError(
            f"Continuation stopped at k={rounds}, not k={TARGET_HORIZON}."
        )
    return result, rounds


def _prefix_preservation(
    *,
    source_result: Mapping[str, Any],
    target_result: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        source = source_result["run"]["accepted_trajectory"]
        target = target_result["run"]["accepted_trajectory"][:SOURCE_HORIZON]
    except (KeyError, TypeError) as exc:
        raise ContinuationError("Accepted trajectory is absent from result.") from exc
    if not isinstance(source, list) or len(source) != SOURCE_HORIZON or len(target) != SOURCE_HORIZON:
        raise ContinuationError("Accepted k30 prefix length drifted.")
    for index, (left, right) in enumerate(zip(source, target), start=1):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ContinuationError("Accepted prefix row is malformed.")
        left_copy = dict(left)
        right_copy = dict(right)
        left_energy = float(left_copy.pop("energy"))
        right_energy = float(right_copy.pop("energy"))
        tolerance = 128.0 * math.ulp(max(1.0, abs(left_energy), abs(right_energy)))
        if left_copy != right_copy or not math.isclose(
            left_energy,
            right_energy,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ContinuationError(
                f"Authenticated accepted-state prefix changed at k={index}."
            )
    terminal = target[-1]
    return {
        "status": "passed",
        "source_round": SOURCE_HORIZON,
        "terminal_energy": float(terminal["energy"]),
        "terminal_state_fingerprint": terminal[
            "projective_state_fingerprint"
        ],
        "all_non_energy_fields_exact": True,
        "energy_comparison": "128_ulp_roundoff_only",
    }


def _quarantine_failure(
    worker: Any,
    *,
    staging: Path,
    runtime_dir: Path,
    execution_id: str,
    failure: BaseException,
) -> Path:
    destination = runtime_dir / "quarantine" / execution_id
    temporary = runtime_dir / "quarantine" / f".{execution_id}.{os.getpid()}.tmp"
    if (
        not staging.is_dir()
        or staging.is_symlink()
        or destination.exists()
        or destination.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise ContinuationError(f"Cannot safely quarantine: {execution_id}")
    os.rename(staging, temporary)
    receipt = _digested(
        {
            "schema": QUARANTINE_SCHEMA,
            "status": "preserved_post_execute_closure_failure",
            "execution_id": execution_id,
            "adapter_sha256": _sha256_file(ADAPTER_PATH),
            "failure_type": type(failure).__name__,
            "failure_message": str(failure),
            "scientific_execution_completed": True,
            "scientific_output_published": (
                runtime_dir / "runs" / execution_id
            ).is_dir(),
            "retry_execution_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json(temporary / "quarantine_receipt.json", receipt, exclusive=True)
    os.rename(temporary, destination)
    return destination


def run_cell(
    *,
    execution_id: str,
    activation_dir: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    worker = k30._load_worker()
    authority, runtime, bundle = _resume_authorization(
        worker,
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
        execution_id=execution_id,
    )
    expected_token = f"{runtime['sha256']}:{execution_id}"
    if os.environ.get(LOCAL_CHILD_TOKEN_ENV) != expected_token:
        raise ContinuationError("Cell execution is supervisor-only.")
    output_dir = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    if any(path.exists() or path.is_symlink() for path in (output_dir, receipt_path)):
        raise ContinuationError(f"Refusing to overwrite cell: {execution_id}")

    jobs = _job_rows_by_id(worker)
    job_path = PACKAGE_DIR / jobs[execution_id]["job_path"]
    job, manifest, source_protocol, problem, temporary = worker._prepare(job_path)
    scientific_execution_completed = False
    staging = Path(temporary.name) / "cell_output"
    try:
        if (
            authority.get("job_spec_sha256") != job.get("sha256")
            or authority.get("source_protocol_sha256")
            != source_protocol.sha256
            or authority.get("route_contract_sha256")
            != source_protocol.route_contract.get("sha256")
            or manifest.get("sha256") != k30.PACKAGE_MANIFEST_CANONICAL_SHA256
        ):
            raise ContinuationError(f"Resume source identity drifted: {execution_id}")
        run_root = Path(str(authority["source_run_root"]))
        gate = k30._load_digested(
            worker,
            K30_RUNTIME_DIR / "plateau_gates" / f"{execution_id}.json",
            label=f"live k30 gate {execution_id}",
        )
        _validate_resume_gate_files(worker, job=job, run_root=run_root, gate=gate)
        if (
            gate.get("extension_decision")
            != "eligible_for_authenticated_resume_to_k50"
            or gate.get("sha256") != authority.get("k30_plateau_gate_sha256")
        ):
            raise ContinuationError(f"Cell gate is not eligible: {execution_id}")
        checkpoint = run_root / str(gate["resume_checkpoint"]["path"])
        if int(job["target_horizon"]) == TARGET_HORIZON:
            protocol = source_protocol
            derivation_kind = "source_authorized_k50_protocol_reused_exactly"
            target_protocol = authority.get("target_protocol")
            if (
                not isinstance(target_protocol, Mapping)
                or target_protocol.get("kind")
                != "source_protocol_reused_exactly"
                or target_protocol.get("source_protocol_sha256")
                != source_protocol.sha256
                or target_protocol.get("target_protocol_sha256")
                != source_protocol.sha256
            ):
                raise ContinuationError(
                    f"Weak target protocol authority drifted: {execution_id}"
                )
        else:
            protocol = _derive_strong_k50_protocol(
                worker,
                job=job,
                source_protocol=source_protocol,
                continuation_bundle_id=CONTINUATION_BUNDLE_ID,
                continuation_bundle_manifest_sha256=bundle["sha256"],
            )
            derivation_kind = "source_locked_sole_horizon_delta_30_to_50"
            target_protocol = authority.get("target_protocol")
            if not isinstance(target_protocol, Mapping):
                raise ContinuationError(
                    f"Strong target protocol authority is absent: {execution_id}"
                )
            materialized = _verify_binding(
                activation_dir,
                target_protocol,
                expected_path=f"protocols/{execution_id}.json",
                label=f"materialized strong target protocol {execution_id}",
                canonical=True,
            )
            if (
                materialized != protocol.to_dict()
                or target_protocol.get("target_protocol_sha256")
                != protocol.sha256
            ):
                raise ContinuationError(
                    f"Strong target protocol rematerialization drifted: {execution_id}"
                )

        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = _execute_resume(
                protocol=protocol,
                problem=problem,
                checkpoint=checkpoint,
                checkpoint_sha256=str(gate["resume_checkpoint"]["sha256"]),
                staging=staging,
            )
            scientific_execution_completed = True
        finally:
            os.chdir(original)

        result_payload = result.to_dict()
        source_result = worker.load_json(
            run_root / "result/result.json",
            label=f"source k30 result {execution_id}",
        )
        prefix = _prefix_preservation(
            source_result=source_result,
            target_result=result_payload,
        )
        worker._write_json(staging / "result/result.json", result_payload)
        if result.run.paper_i_summary is None:
            raise ContinuationError(f"Continuation summary is absent: {execution_id}")
        k30._write_summary_for_validation(
            worker,
            staging / "summary/summary.json",
            result.run.paper_i_summary,
        )
        worker._write_json(
            staging / "continuation/resolved_protocol.json",
            protocol.to_dict(),
        )
        worker._write_json(
            staging / "continuation/resume_authorization.json",
            authority,
        )
        source_audit = _digested(
            {
                "schema": "paper_i_page16_k30_to_k50_source_lock_audit_v2",
                "status": "passed",
                "execution_id": execution_id,
                "source_protocol_sha256": source_protocol.sha256,
                "target_protocol_sha256": protocol.sha256,
                "common_route_contract_sha256": protocol.route_contract["sha256"],
                "comparator_policy": job["comparator_policy"],
                "source_horizon": int(job["target_horizon"]),
                "resume_round": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "protocol_derivation_kind": derivation_kind,
                "non_horizon_protocol_diff": [],
                "source_locks_exact": protocol.source_locks
                == source_protocol.source_locks,
                "resume_checkpoint_sha256": gate["resume_checkpoint"]["sha256"],
                "resume_checkpoint_siblings": gate[
                    "resume_checkpoint_siblings"
                ],
                "accepted_prefix_preservation": prefix,
            }
        )
        worker._write_json(
            staging / "continuation/source_lock_audit.json",
            source_audit,
        )
        payloads = {
            relative.as_posix(): {
                "sha256": worker.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file() and not path.is_symlink()
            for relative in [path.relative_to(staging)]
        }
        execution_manifest = _digested(
            {
                "schema": EXECUTION_SCHEMA,
                "status": "passed",
                "execution_target": LOCAL_EXECUTION_TARGET,
                "source_package_id": manifest["package_id"],
                "source_package_manifest_sha256": manifest["sha256"],
                "adapter_sha256": runtime["adapter_sha256"],
                "activation_manifest_sha256": runtime[
                    "activation_manifest_sha256"
                ],
                "execution_id": execution_id,
                "job_spec_sha256": job["sha256"],
                "resume_authorization_sha256": authority["sha256"],
                "source_protocol_sha256": source_protocol.sha256,
                "protocol_sha256": protocol.sha256,
                "protocol_derivation_kind": derivation_kind,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "comparator_policy": job["comparator_policy"],
                "resume_round": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "controller_rounds_completed": rounds,
                "source_checkpoint_sha256": gate["resume_checkpoint"]["sha256"],
                "source_plateau_gate_sha256": gate["sha256"],
                "accepted_state_resume": True,
                "fresh_start": False,
                "accepted_prefix_preservation": prefix,
                "source_lock_audit_sha256": source_audit["sha256"],
                "output_payloads": payloads,
                "paper_evidence_adoption_authorized": False,
            }
        )
        worker._write_json(staging / "execution_manifest.json", execution_manifest)
        worker._publish_staging(staging, output_dir)
        receipt = _digested(
            {
                "schema": WORKER_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_id": execution_id,
                "job_spec_sha256": job["sha256"],
                "resume_authorization_sha256": authority["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "resume_round": SOURCE_HORIZON,
                "controller_rounds_completed": rounds,
                "accepted_state_resume": True,
                "fresh_start": False,
                "artifacts": [
                    {
                        "path": (
                            PurePosixPath("runs")
                            / execution_id
                            / path.relative_to(output_dir)
                        ).as_posix(),
                        "sha256": worker.sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(output_dir.rglob("*"))
                    if path.is_file() and not path.is_symlink()
                ],
            }
        )
        _write_json(receipt_path, receipt, exclusive=True)
        return receipt
    except BaseException as exc:
        if scientific_execution_completed and staging.is_dir():
            _quarantine_failure(
                worker,
                staging=staging,
                runtime_dir=runtime_dir,
                execution_id=execution_id,
                failure=exc,
            )
        raise
    finally:
        temporary.cleanup()


def closed_continuation_cell(
    *,
    runtime_dir: Path,
    execution_id: str,
) -> bool:
    run_root = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    if not any(path.exists() or path.is_symlink() for path in (run_root, receipt_path)):
        return False
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise ContinuationError(f"Partial continuation output: {execution_id}")
    manifest = _load_digested(
        run_root / "execution_manifest.json",
        label=f"continuation execution manifest {execution_id}",
    )
    receipt = _load_digested(
        receipt_path,
        label=f"continuation worker receipt {execution_id}",
    )
    if (
        manifest.get("schema") != EXECUTION_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("resume_round") != SOURCE_HORIZON
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or manifest.get("accepted_state_resume") is not True
        or manifest.get("fresh_start") is not False
        or manifest.get("accepted_prefix_preservation", {}).get("status")
        != "passed"
        or receipt.get("schema") != WORKER_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or receipt.get("controller_rounds_completed") != TARGET_HORIZON
        or receipt.get("accepted_state_resume") is not True
    ):
        raise ContinuationError(f"Continuation closure drifted: {execution_id}")
    k30._verify_receipt_artifacts(k30._load_worker(), runtime_dir, receipt)
    return True


def _main_payload(value: Mapping[str, Any]) -> None:
    print(_canonical_json_bytes(value).decode("utf-8"), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conditional authenticated Page-16 k30 to k50 adapter"
    )
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--decision-status", action="store_true")
    parser.add_argument("--run-cell", choices=CONDITIONAL_EXECUTION_IDS)
    args = parser.parse_args()
    choices = [args.prepare, args.preflight, args.decision_status, args.run_cell is not None]
    if sum(bool(row) for row in choices) != 1:
        parser.error("choose exactly one action")
    activation_dir = args.activation_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    try:
        if args.prepare:
            _main_payload(prepare_activation(activation_dir=activation_dir))
            return 0
        if args.preflight:
            _main_payload(
                inert_preflight(
                    activation_dir=activation_dir,
                    runtime_dir=runtime_dir,
                )
            )
            return 0
        if args.decision_status:
            _main_payload(decision_snapshot())
            return 0
        assert args.run_cell is not None
        _main_payload(
            run_cell(
                execution_id=args.run_cell,
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
            )
        )
        return 0
    except (ContinuationError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
