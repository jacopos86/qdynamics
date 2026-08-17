#!/usr/bin/env python3
"""Authorize and run the local Page-12 matched singleton-12 campaign.

This runner is local-only, handoff-gated, serial, and archive-backed.  It does
not submit work and it never authorizes Paper-I evidence adoption.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import psutil


RUNNER_PATH = Path(__file__).resolve()
REPAIR_ROOT = RUNNER_PATH.parent
REPO_ROOT = RUNNER_PATH.parents[2]
PYTHON_EXECUTABLE = Path(
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
)
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_page12_matched_singleton12_r50_20260815_v1_local"
)
WORKER_PATH = PACKAGE_DIR / "run_cell.py"
PACKAGE_CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
)
ARCHIVE_MODULE_PATH = REPAIR_ROOT / (
    "paper_i_matched_singleton12_archive_20260815.py"
)
GATE_PATH = REPAIR_ROOT / (
    "handoff_local_page12_strong5_to_matched_singleton12_20260815.py"
)
DEFAULT_PLANNING_DIR = REPAIR_ROOT / (
    "paper_i_page12_matched_singleton12_r50_20260815_v1_planning"
)
DEFAULT_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_page12_matched_singleton12_r50_20260815_v1_activation"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/paper_i_page12_matched_singleton12_r50_20260815_v1"
)
DEFAULT_TARGET_CONTRACT = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_target_contract_20260815.json"
)
DEFAULT_HANDOFF_STATE_DIR = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_handoff_state_20260815_v1"
)
DEFAULT_HANDOFF_RECEIPT = DEFAULT_HANDOFF_STATE_DIR / "handoff_receipt.json"
DEFAULT_HANDOFF_LOCK = DEFAULT_HANDOFF_STATE_DIR / "handoff.lock"

PACKAGE_MANIFEST_SHA256 = (
    "e1ef531040468faf1f4524d63a2b0b6703b94747fd685b4707b65c51b8bc40b5"
)
SEALED_RESUME_READER_SHA256 = (
    "173fcbc219453b4a90d604afdfe117718a34318bc621a11ab178a63304e72032"
)
EXECUTION_SOURCE_POLICY = (
    "sealed_archive_plus_single_authorized_post_extraction_overlay"
)
CHECKPOINT_USAGE = "compact_observation_only"
PARITY_CANARY_SCOPE = "one_round_scientific_and_ledger_equivalence"
TARGET_HORIZON = 50
MAXIMUM_CONCURRENCY = 1
RSS_LIMIT_BYTES = 12 * 1024**3
MINIMUM_AVAILABLE_MEMORY_BYTES = 2 * 1024**3
MINIMUM_FREE_DISK_BYTES = 31 * 1024**3
RUNTIME_FREE_DISK_FLOOR_BYTES = 2 * 1024**3
GUARD_POLL_SECONDS = 1.0

REQUIRED_NUMERICAL_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
}

HANDOFF_RECEIPT_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_RECEIPT"
HANDOFF_TOKEN_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_TOKEN"
HANDOFF_LOCK_FD_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_LOCK_FD"
CHILD_TOKEN_ENV = "PAPER_I_MATCHED_SINGLETON12_CHILD_TOKEN"

PLAN_SCHEMA = "paper_i_page12_matched_singleton12_local_execution_plan_v1"
PLANNING_SCHEMA = "paper_i_page12_matched_singleton12_local_planning_manifest_v1"
AUTHORIZATION_SCHEMA = "paper_i_page12_matched_singleton12_local_execution_authorization_v1"
ACTIVATION_SCHEMA = "paper_i_page12_matched_singleton12_local_activation_v1"
PARITY_SCHEMA = "paper_i_page12_matched_singleton12_scientific_parity_canary_v1"
PAIR_PARITY_SCHEMA = "paper_i_page12_matched_singleton12_native_runtime_pair_parity_v1"
RUNTIME_FINGERPRINT_SCHEMA = "paper_i_matched_singleton12_live_runtime_fingerprint_v1"
RUNTIME_SCHEMA = "paper_i_page12_matched_singleton12_local_runtime_v1"
STATUS_SCHEMA = "paper_i_page12_matched_singleton12_local_status_v1"
RUNTIME_CHECK_SCHEMA = "paper_i_page12_matched_singleton12_runtime_check_v1"
TERMINAL_SCHEMA = "paper_i_page12_matched_singleton12_local_terminal_receipt_v1"
TERMINAL_STATUS = "passed_all_twelve_cells_immutable_archive_closure"
TARGET_CONTRACT_SCHEMA = "paper_i_matched_singleton12_handoff_contract_v1"
TARGET_CONTRACT_STATUS = "passed_target_campaign_preauthorized"
HANDOFF_RECEIPT_SCHEMA = "paper_i_matched_singleton12_handoff_receipt_v1"
HANDOFF_RECEIPT_STATUS = (
    "passed_source_terminal_and_target_activation_authorized_pending_exec"
)
PARITY_STATUS = "passed_matched_protocol_identity_and_checkpoint_overlay_parity"

class MatchedSingleton12Error(RuntimeError):
    """A fail-closed local matched-suite contract failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MatchedSingleton12Error(f"{label} is absent or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MatchedSingleton12Error(f"{label} must be a JSON object.")
    if payload.get("sha256") != _canonical_sha256(
        {key: item for key, item in payload.items() if key != "sha256"}
    ):
        raise MatchedSingleton12Error(f"{label} self-digest drifted.")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise MatchedSingleton12Error(f"Stale atomic temporary: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic_noreplace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.publish")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(path)
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _binding(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MatchedSingleton12Error(f"Cannot bind absent or unsafe file: {path}")
    result: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
    }
    if canonical:
        result["canonical_sha256"] = _load_digested(path, label=path.name)["sha256"]
    return result


def _staged_binding(
    staged_path: Path, published_path: Path, *, canonical: bool = False
) -> dict[str, Any]:
    """Bind staged bytes to the path they will have after atomic publication."""

    binding = _binding(staged_path, canonical=canonical)
    return {**binding, "path": published_path.resolve().as_posix()}


def _validate_binding(raw: Any, *, label: str, root: Path | None = None) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(raw, Mapping):
        raise MatchedSingleton12Error(f"{label} binding is malformed.")
    path = Path(str(raw.get("path", "")))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MatchedSingleton12Error(f"{label} binding path is unsafe.")
    if root is not None:
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise MatchedSingleton12Error(f"{label} binding escaped authority root.") from exc
    if path.stat().st_size != int(raw.get("size_bytes", -1)) or _sha256_file(path) != raw.get("file_sha256"):
        raise MatchedSingleton12Error(f"{label} byte binding drifted.")
    payload = None
    if "canonical_sha256" in raw:
        payload = _load_digested(path, label=label)
        if payload["sha256"] != raw.get("canonical_sha256"):
            raise MatchedSingleton12Error(f"{label} canonical binding drifted.")
    return path, payload


def _live_runtime_fingerprint() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "qiskit", "qiskit-aer", "psutil"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    executable = Path(sys.executable)
    return _digested(
        {
            "schema": RUNTIME_FINGERPRINT_SCHEMA,
            "python_executable": executable.as_posix(),
            "python_executable_resolved": executable.resolve().as_posix(),
            "python_executable_sha256": _sha256_file(executable.resolve()),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "packages": packages,
            "numerical_environment": {
                key: os.environ.get(key) for key in REQUIRED_NUMERICAL_ENVIRONMENT
            },
        }
    )


def _capacity(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(probe)
    return _digested(
        {
            "schema": "paper_i_page12_matched_singleton12_capacity_v1",
            "observed_at_utc": _utc_now(),
            "probe_path": probe.resolve().as_posix(),
            "available_memory_bytes": int(memory.available),
            "free_disk_bytes": int(disk.free),
        }
    )


def _existing_capacity_probe(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists() or probe.is_symlink():
        raise MatchedSingleton12Error("Capacity probe path is absent or unsafe.")
    return probe


def _load_module(path: Path, name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise MatchedSingleton12Error(f"Required module is absent or unsafe: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MatchedSingleton12Error(f"Cannot load required module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _package_manifest() -> dict[str, Any]:
    manifest = _load_digested(PACKAGE_DIR / "package_manifest.json", label="matched package manifest")
    overlay = manifest.get("operational_checkpoint_overlay", {})
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or manifest.get("status") != "passed_inert_matched_singleton12"
        or manifest.get("row_count") != 12
        or manifest.get("methods") != ["ra_singleton_plateau", "append_singleton"]
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("paper_adoption_authorized") is not False
        or manifest.get("paper_evidence_adoption_authorized") is not False
        or not isinstance(overlay, Mapping)
        or overlay.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or overlay.get("post_extraction_overlay_count") != 1
        or overlay.get("ambient_resume_overlay") is not False
        or overlay.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or overlay.get("fresh_start_only") is not True
        or overlay.get("checkpoint_usage") != CHECKPOINT_USAGE
        or overlay.get("checkpoint_resume_authorized") is not False
        or overlay.get("parity_canary_scope") != PARITY_CANARY_SCOPE
        or overlay.get("multi_round_compact_tail_resume_validated") is not False
    ):
        raise MatchedSingleton12Error("Matched package manifest drifted.")
    _validate_package_control_files(manifest)
    return manifest


def _validate_package_control_files(manifest: Mapping[str, Any]) -> None:
    raw_rows = manifest.get("control_files")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(PACKAGE_CONTROL_FILES):
        raise MatchedSingleton12Error("Matched package control file closure drifted.")
    rows: list[Mapping[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise MatchedSingleton12Error("Matched package control file binding is malformed.")
        rows.append(raw)
    if [row.get("path") for row in rows] != list(PACKAGE_CONTROL_FILES):
        raise MatchedSingleton12Error("Matched package control file inventory drifted.")
    for row in rows:
        relative = str(row["path"])
        path = PACKAGE_DIR / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(PACKAGE_DIR.resolve(strict=True))
            expected_size = int(row["size_bytes"])
        except (FileNotFoundError, TypeError, ValueError) as exc:
            raise MatchedSingleton12Error(
                f"Matched package control file path is unsafe: {relative}"
            ) from exc
        if (
            path.is_symlink()
            or not path.is_file()
            or resolved != path.absolute()
            or path.stat().st_size != expected_size
            or _sha256_file(path) != row["sha256"]
        ):
            raise MatchedSingleton12Error(
                f"Matched package control file byte binding drifted: {relative}"
            )


def _cell_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    order = manifest.get("execution_order")
    if not isinstance(jobs, list) or not isinstance(order, list) or len(jobs) != 12 or len(order) != 12:
        raise MatchedSingleton12Error("Matched job/order closure drifted.")
    by_execution: dict[str, dict[str, Any]] = {}
    for raw in jobs:
        if not isinstance(raw, Mapping):
            raise MatchedSingleton12Error("Matched job binding is malformed.")
        path = PACKAGE_DIR / str(raw.get("path", ""))
        job = _load_digested(path, label="matched job")
        if _sha256_file(path) != raw.get("sha256") or job["sha256"] != raw.get("canonical_sha256"):
            raise MatchedSingleton12Error("Matched job binding drifted.")
        if (
            job.get("fresh_start_contract", {}).get("fresh_start_only")
            is not True
            or job.get("paper_adoption_authorized") is not False
            or job.get("paper_evidence_adoption_authorized") is not False
            or job.get("fresh_start_contract", {}).get(
                "checkpoint_resume_authorized"
            )
            is not False
            or job.get("checkpoint_observation", {}).get("usage")
            != CHECKPOINT_USAGE
            or job.get("checkpoint_observation", {}).get("resume_consumable")
            is not False
        ):
            raise MatchedSingleton12Error("Matched checkpoint authority drifted.")
        execution = str(job["execution_id"])
        if execution in by_execution:
            raise MatchedSingleton12Error("Matched execution ID is duplicated.")
        by_execution[execution] = {
            "execution_id": execution,
            "method": str(job["method"]),
            "regime": str(job["regime_id"]),
            "n_ph": int(job["nph"]),
            "job_path": path.resolve().as_posix(),
            "job_spec_sha256": job["sha256"],
            "protocol_sha256": job["protocol_sha256"],
            "route_contract_sha256": job["route_contract_sha256"],
            "max_runtime_seconds": int(job["resources"]["max_runtime_seconds"]),
        }
    if set(by_execution) != set(order):
        raise MatchedSingleton12Error("Matched execution order is not closed.")
    rows = [by_execution[str(execution)] for execution in order]
    expected_pairs = [
        ("strong_strong_u8", 7),
        ("intermediate_strong", 7),
        ("weak_strong", 7),
        ("strong_weak_u8", 3),
        ("intermediate_weak", 3),
        ("weak_weak", 3),
    ]
    observed_pairs = [(rows[index]["regime"], rows[index]["n_ph"]) for index in range(0, 12, 2)]
    if (
        observed_pairs != expected_pairs
        or any(rows[index]["method"] != "ra_singleton_plateau" for index in range(0, 12, 2))
        or any(rows[index]["method"] != "append_singleton" for index in range(1, 12, 2))
    ):
        raise MatchedSingleton12Error("Strongest-first matched-pair order drifted.")
    return rows


def _archive_module() -> Any:
    module = _load_module(
        ARCHIVE_MODULE_PATH, "paper_i_matched_singleton12_archive_runtime"
    )
    required = (
        "ArchiveLimits",
        "CellArchivePaths",
        "build_cell_archive",
        "publish_archive_closure",
        "publish_rotation_intent",
        "complete_safe_tree_rotation",
        "validate_archive_backed_closure",
        "inspect_rotation_state",
        "discard_stale_archive_temporaries",
        "require_campaign_capacity",
        "require_regime_launch_capacity",
        "campaign_default_archive_limits",
        "campaign_capacity_floor",
        "campaign_archive_capacity_contract",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        raise MatchedSingleton12Error("Archive module API closure drifted.")
    for name in (
        "PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_BYTES",
        "PAPER_I_MATCHED_SINGLETON12_ARCHIVE_MAX_COMPRESSED_BYTES",
        "PAPER_I_MATCHED_SINGLETON12_ARCHIVE_POST_WRITE_RESERVE_BYTES",
    ):
        if not isinstance(getattr(module, name, None), int):
            raise MatchedSingleton12Error("Archive capacity constants drifted.")
    return module


def _archive_limits(module: Any) -> Any:
    limits = module.campaign_default_archive_limits()
    if not isinstance(limits, module.ArchiveLimits):
        raise MatchedSingleton12Error("Archive default limits type drifted.")
    return limits


def prepare_planning(*, planning_dir: Path = DEFAULT_PLANNING_DIR, runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    if planning_dir.exists() or planning_dir.is_symlink():
        raise FileExistsError(planning_dir)
    manifest = _package_manifest()
    cells = _cell_rows(manifest)
    archive = _archive_module()
    limits = _archive_limits(archive)
    worker = _load_module(WORKER_PATH, "paper_i_page12_matched12_worker_planning")
    # Exercise both facades and both cutoff sectors without scientific work.
    preflight_indices = (0, 1, 10, 11)
    preflights = [worker.preflight(Path(cells[index]["job_path"])) for index in preflight_indices]
    if any(
        item.get("status") != "passed"
        or item.get("scientific_execution_performed") is not False
        or item.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or item.get("fresh_start_only") is not True
        or item.get("checkpoint_usage") != CHECKPOINT_USAGE
        or item.get("checkpoint_resume_authorized") is not False
        or item.get("checkpoint_overlay", {}).get("ambient_resume_overlay")
        is not False
        or item.get("checkpoint_overlay", {}).get(
            "sealed_resume_reader_sha256"
        )
        != SEALED_RESUME_READER_SHA256
        for item in preflights
    ):
        raise MatchedSingleton12Error("Worker preflight canary failed.")
    capacity = _capacity(runtime_dir)
    archive_capacity = archive.require_campaign_capacity(
        _existing_capacity_probe(runtime_dir)
    )
    if (
        int(capacity["available_memory_bytes"]) < MINIMUM_AVAILABLE_MEMORY_BYTES
    ):
        raise MatchedSingleton12Error("Planning capacity floor is not satisfied.")
    plan = _digested(
        {
            "schema": PLAN_SCHEMA,
            "status": "passed_strongest_pairs_first_serial_plan",
            "package_manifest_sha256": manifest["sha256"],
            "target_horizon": TARGET_HORIZON,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "cells": cells,
            "execution_ids": [row["execution_id"] for row in cells],
            "compact_checkpoint_keep_history_tail": 1,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "same_native_runtime_per_matched_pair": True,
            "archive_rotation_required_after_each_cell": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    preflight_receipt = _digested(
        {
            "schema": "paper_i_page12_matched_singleton12_planning_preflight_v1",
            "status": "passed_inert_preflight",
            "worker_preflights": preflights,
            "archive_module_api_present": True,
            "archive_limits": {
                **limits.as_dict(),
                "archive_start_free_floor_bytes": (
                    limits.archive_start_free_floor_bytes
                ),
            },
            "capacity": capacity,
            "archive_campaign_capacity": archive_capacity,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "scientific_execution_performed": False,
        }
    )
    staging_dir = planning_dir.with_name(f".{planning_dir.name}.in_progress")
    if staging_dir.exists() or staging_dir.is_symlink():
        if staging_dir.is_symlink() or not staging_dir.is_dir():
            raise MatchedSingleton12Error(
                "Planning staging path is unsafe."
            )
        quarantine = planning_dir.with_name(
            f".{planning_dir.name}.quarantine.{time.time_ns()}"
        )
        os.rename(staging_dir, quarantine)
        _fsync_directory(planning_dir.parent)
    staging_dir.mkdir(parents=True, exist_ok=False)
    staged_plan_path = staging_dir / "execution_plan.json"
    staged_preflight_path = staging_dir / "inert_preflight.json"
    staged_planning_path = staging_dir / "planning_manifest.json"
    _write_json_exclusive(staged_plan_path, plan)
    _write_json_exclusive(staged_preflight_path, preflight_receipt)
    planning = _digested(
        {
            "schema": PLANNING_SCHEMA,
            "status": "passed_inert_planning",
            "created_at_utc": _utc_now(),
            "repo_root": REPO_ROOT.resolve().as_posix(),
            "runtime_dir": runtime_dir.resolve().as_posix(),
            "package_manifest": _binding(PACKAGE_DIR / "package_manifest.json", canonical=True),
            "runner": _binding(RUNNER_PATH),
            "worker": _binding(WORKER_PATH),
            "archive_module": _binding(ARCHIVE_MODULE_PATH),
            "execution_plan": _staged_binding(
                staged_plan_path,
                planning_dir / "execution_plan.json",
                canonical=True,
            ),
            "inert_preflight": _staged_binding(
                staged_preflight_path,
                planning_dir / "inert_preflight.json",
                canonical=True,
            ),
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(staged_planning_path, planning)
    if (
        _load_digested(staged_plan_path, label="staged execution plan") != plan
        or _load_digested(
            staged_preflight_path, label="staged inert preflight"
        )
        != preflight_receipt
        or _load_digested(
            staged_planning_path, label="staged planning manifest"
        )
        != planning
        or planning.get("execution_plan")
        != _staged_binding(
            staged_plan_path,
            planning_dir / "execution_plan.json",
            canonical=True,
        )
        or planning.get("inert_preflight")
        != _staged_binding(
            staged_preflight_path,
            planning_dir / "inert_preflight.json",
            canonical=True,
        )
    ):
        raise MatchedSingleton12Error("Staged planning closure drifted.")
    _fsync_directory(staging_dir)
    if planning_dir.exists() or planning_dir.is_symlink():
        raise FileExistsError(planning_dir)
    os.rename(staging_dir, planning_dir)
    _fsync_directory(planning_dir.parent)
    validated_planning, validated_plan = _validated_planning(planning_dir)
    if validated_planning != planning or validated_plan != plan:
        raise MatchedSingleton12Error(
            "Published planning closure drifted."
        )
    return planning


def _validated_planning(planning_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    planning = _load_digested(planning_dir / "planning_manifest.json", label="planning manifest")
    plan = _load_digested(planning_dir / "execution_plan.json", label="execution plan")
    preflight = _load_digested(
        planning_dir / "inert_preflight.json", label="planning inert preflight"
    )
    manifest = _package_manifest()
    expected_cells = _cell_rows(manifest)
    if (
        planning.get("schema") != PLANNING_SCHEMA
        or planning.get("status") != "passed_inert_planning"
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "passed_strongest_pairs_first_serial_plan"
        or plan.get("package_manifest_sha256") != manifest["sha256"]
        or plan.get("cells") != expected_cells
        or plan.get("execution_ids")
        != [cell["execution_id"] for cell in expected_cells]
        or plan.get("target_horizon") != TARGET_HORIZON
        or plan.get("maximum_concurrency") != 1
        or plan.get("compact_checkpoint_keep_history_tail") != 1
        or plan.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or plan.get("fresh_start_only") is not True
        or plan.get("checkpoint_usage") != CHECKPOINT_USAGE
        or plan.get("checkpoint_resume_authorized") is not False
        or plan.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or planning.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or planning.get("fresh_start_only") is not True
        or planning.get("checkpoint_usage") != CHECKPOINT_USAGE
        or planning.get("checkpoint_resume_authorized") is not False
        or planning.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or planning.get("execution_authorized") is not False
        or plan.get("execution_authorized") is not False
        or planning.get("submission_authorized") is not False
        or plan.get("submission_authorized") is not False
        or planning.get("paper_adoption_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or planning.get("paper_evidence_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
        or preflight.get("status") != "passed_inert_preflight"
        or preflight.get("scientific_execution_performed") is not False
        or preflight.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or preflight.get("fresh_start_only") is not True
        or preflight.get("checkpoint_usage") != CHECKPOINT_USAGE
        or preflight.get("checkpoint_resume_authorized") is not False
        or preflight.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or planning.get("runner") != _binding(RUNNER_PATH)
        or planning.get("worker") != _binding(WORKER_PATH)
        or planning.get("archive_module") != _binding(ARCHIVE_MODULE_PATH)
        or planning.get("package_manifest") != _binding(PACKAGE_DIR / "package_manifest.json", canonical=True)
        or planning.get("execution_plan") != _binding(planning_dir / "execution_plan.json", canonical=True)
        or planning.get("inert_preflight")
        != _binding(planning_dir / "inert_preflight.json", canonical=True)
    ):
        raise MatchedSingleton12Error("Planning authority drifted.")
    return planning, plan


def _publish_target_contract(
    *, activation_dir: Path, plan: Mapping[str, Any], target_contract_path: Path
) -> dict[str, Any]:
    if target_contract_path.exists() or target_contract_path.is_symlink():
        raise FileExistsError(target_contract_path)
    archive = _archive_module()
    minimum_disk = int(
        archive.PAPER_I_MATCHED_SINGLETON12_CAPACITY_FLOOR_BYTES
    )
    if minimum_disk != MINIMUM_FREE_DISK_BYTES:
        raise MatchedSingleton12Error("Target/archive campaign floor drifted.")
    bindings = {
        "planning_manifest": _binding(activation_dir / "planning_manifest.json", canonical=True),
        "execution_plan": _binding(activation_dir / "execution_plan.json", canonical=True),
        "execution_authorization": _binding(activation_dir / "execution_authorization.json", canonical=True),
        "activation_manifest": _binding(activation_dir / "activation_manifest.json", canonical=True),
        "scientific_parity_canary": _binding(activation_dir / "scientific_parity_canary.json", canonical=True),
        "runtime_fingerprint": _binding(activation_dir / "runtime_fingerprint.json", canonical=True),
    }
    source_runner = REPAIR_ROOT / "run_local_page12_strong_holstein_sector5_20260814.py"
    source_activation_dir = REPAIR_ROOT / "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1_activation"
    source_runtime_dir = REPO_ROOT / "output/local_runs/paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1"
    cells = [
        {
            "execution_id": row["execution_id"],
            "method": row["method"],
            "regime": row["regime"],
            "n_ph": row["n_ph"],
        }
        for row in plan["cells"]
    ]
    target = {
        "repo_root": REPO_ROOT.resolve().as_posix(),
        "runner_path": RUNNER_PATH.as_posix(),
        "runner_sha256": _sha256_file(RUNNER_PATH),
        "activation_dir": activation_dir.resolve().as_posix(),
        "runtime_dir": DEFAULT_RUNTIME_DIR.resolve().as_posix(),
        "command": [PYTHON_EXECUTABLE.as_posix(), "-B", RUNNER_PATH.as_posix(), "--run-campaign"],
        "environment": REQUIRED_NUMERICAL_ENVIRONMENT,
        "maximum_concurrency": 1,
        "minimum_free_disk_bytes": minimum_disk,
        "minimum_available_memory_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
        "execution_source_policy": EXECUTION_SOURCE_POLICY,
        "fresh_start_only": True,
        "checkpoint_usage": CHECKPOINT_USAGE,
        "checkpoint_resume_authorized": False,
        "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
        "cells": cells,
        "authority_bindings": bindings,
        "expected_parity_status": PARITY_STATUS,
        "expected_terminal": {
            "path": (DEFAULT_RUNTIME_DIR / "terminal_receipt.json").resolve().as_posix(),
            "schema": TERMINAL_SCHEMA,
            "status": TERMINAL_STATUS,
        },
        "handoff_receipt_environment_variable": HANDOFF_RECEIPT_ENV,
        "handoff_token_environment_variable": HANDOFF_TOKEN_ENV,
        "handoff_lock_fd_environment_variable": HANDOFF_LOCK_FD_ENV,
        "scientific_overlap_markers": [
            [RUNNER_PATH.as_posix(), "--run-campaign"],
            [RUNNER_PATH.as_posix(), "--child-cell"],
        ],
    }
    contract = _digested(
        {
            "schema": TARGET_CONTRACT_SCHEMA,
            "status": TARGET_CONTRACT_STATUS,
            "created_at_utc": _utc_now(),
            "gate_script_path": GATE_PATH.as_posix(),
            "gate_script_sha256": _sha256_file(GATE_PATH),
            "source_prerequisite": {
                "runner_path": source_runner.as_posix(),
                "runner_sha256": "d0e20540f0217364adc47df2c90a8f594469c70f99d32ad280dfd95c2482d8cb",
                "activation_dir": source_activation_dir.as_posix(),
                "activation_manifest_sha256": "7b0851d108eeb15e5285df6c3745fa85befe25cec122811a338321a7b9b94518",
                "runtime_dir": source_runtime_dir.as_posix(),
                "runtime_manifest_sha256": "0f7697f0d4cda4d74705339138b668489038df2519e66a5035bfc776f834b031",
                "terminal_schema": "paper_i_page12_strong_sector5_local_terminal_receipt_v1",
                "terminal_status": "passed_all_five_cells_immutable_closure",
                "final_status": "passed_all_five_cells",
                "execution_ids": [
                    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__strong_strong_u8__nph7__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_always_commutation_reduced",
                    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__strong_strong_u8__nph7__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_append_only",
                    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__intermediate_strong__nph7__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_always_commutation_reduced",
                    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__intermediate_strong__nph7__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_append_only",
                    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__weak_strong__nph7__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_always_commutation_reduced",
                ],
            },
            "target": target,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_atomic_noreplace(target_contract_path, contract)
    return contract


def authorize_activation(
    *,
    planning_dir: Path = DEFAULT_PLANNING_DIR,
    activation_dir: Path = DEFAULT_ACTIVATION_DIR,
    target_contract_path: Path = DEFAULT_TARGET_CONTRACT,
) -> dict[str, Any]:
    if target_contract_path.exists() or target_contract_path.is_symlink():
        raise FileExistsError(target_contract_path)
    if activation_dir.exists() or activation_dir.is_symlink():
        activation, existing_plan, _authorization, _runtime = (
            _validate_activation(activation_dir)
        )
        _publish_target_contract(
            activation_dir=activation_dir,
            plan=existing_plan,
            target_contract_path=target_contract_path,
        )
        return activation
    planning, plan = _validated_planning(planning_dir)
    manifest = _package_manifest()
    runtime_fingerprint = _live_runtime_fingerprint()
    if runtime_fingerprint["numerical_environment"] != REQUIRED_NUMERICAL_ENVIRONMENT:
        raise MatchedSingleton12Error("Activation numerical environment is not exact.")
    strong5_parity_path = REPAIR_ROOT / (
        "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1_activation/"
        "scientific_parity_canary.json"
    )
    strong5_parity = _load_digested(strong5_parity_path, label="checkpoint parity canary")
    if (
        strong5_parity.get("status") != "passed_exact_scientific_parity"
        or strong5_parity.get("sha256")
        != "ad870ca15fd75b31400986c71245a56283532a5b5714b1c456185ce87ad0ceaa"
        or _sha256_file(strong5_parity_path)
        != "ecd8eec182cc9110f35f6ffb8417d3c9d3c97a4d3b07184046b56396ecc1c6ee"
    ):
        raise MatchedSingleton12Error("Checkpoint parity canary did not pass exactly.")
    staging_dir = activation_dir.with_name(f".{activation_dir.name}.in_progress")
    if staging_dir.exists() or staging_dir.is_symlink():
        quarantine = activation_dir.with_name(
            f".{activation_dir.name}.quarantine.{int(time.time())}"
        )
        os.rename(staging_dir, quarantine)
    staging_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(staging_dir / "planning_manifest.json", planning)
    _write_json_exclusive(staging_dir / "execution_plan.json", plan)
    _write_json_exclusive(staging_dir / "runtime_fingerprint.json", runtime_fingerprint)
    parity = _digested(
        {
            "schema": PARITY_SCHEMA,
            "status": PARITY_STATUS,
            "package_manifest_sha256": manifest["sha256"],
            "matched_bundle_validation": manifest["bundle_validation_report"],
            "sealed_ra_protocol_sha256_by_regime": {
                row["regime"]: row["protocol_sha256"]
                for row in plan["cells"]
                if row["method"] == "ra_singleton_plateau"
            },
            "checkpoint_overlay_parity_canary": _binding(strong5_parity_path, canonical=True),
            "checkpoint_overlay_parity_scope": PARITY_CANARY_SCOPE,
            "multi_round_compact_tail_resume_validated": False,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "same_resolved_problem_and_pool_per_pair": True,
            "append_conventional_unwhitened": True,
            "scientific_execution_performed": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(staging_dir / "scientific_parity_canary.json", parity)
    authorization = _digested(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_local_matched_singleton12_execution",
            "authorized_at_utc": _utc_now(),
            "authorized_by": "user_confirmed_local_paper_i_singleton_rerun",
            "planning_manifest_sha256": planning["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "package_manifest_sha256": manifest["sha256"],
            "runtime_fingerprint_sha256": runtime_fingerprint["sha256"],
            "scientific_parity_canary_sha256": parity["sha256"],
            "execution_ids": plan["execution_ids"],
            "maximum_concurrency": 1,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(staging_dir / "execution_authorization.json", authorization)
    activation = _digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "status": "authorized_local_execution",
            "activated_at_utc": _utc_now(),
            "package_manifest_sha256": manifest["sha256"],
            "planning_manifest_sha256": planning["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "execution_authorization_sha256": authorization["sha256"],
            "scientific_parity_canary_sha256": parity["sha256"],
            "runtime_fingerprint_sha256": runtime_fingerprint["sha256"],
            "runner": _binding(RUNNER_PATH),
            "worker": _binding(WORKER_PATH),
            "archive_module": _binding(ARCHIVE_MODULE_PATH),
            "execution_ids": plan["execution_ids"],
            "maximum_concurrency": 1,
            "compact_checkpoint_keep_history_tail": 1,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(staging_dir / "activation_manifest.json", activation)
    validated_activation, _validated_plan, _validated_authorization, _validated_runtime = (
        _validate_activation(staging_dir)
    )
    if validated_activation != activation or _validated_plan != plan:
        raise MatchedSingleton12Error("Staged activation validation drifted.")
    _fsync_directory(staging_dir)
    if activation_dir.exists() or activation_dir.is_symlink():
        raise FileExistsError(activation_dir)
    os.rename(staging_dir, activation_dir)
    _fsync_directory(activation_dir.parent)
    _publish_target_contract(
        activation_dir=activation_dir,
        plan=plan,
        target_contract_path=target_contract_path,
    )
    return activation


def _validate_activation(activation_dir: Path = DEFAULT_ACTIVATION_DIR) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    planning = _load_digested(activation_dir / "planning_manifest.json", label="activation planning")
    plan = _load_digested(activation_dir / "execution_plan.json", label="activation plan")
    authorization = _load_digested(activation_dir / "execution_authorization.json", label="execution authorization")
    activation = _load_digested(activation_dir / "activation_manifest.json", label="activation manifest")
    parity = _load_digested(activation_dir / "scientific_parity_canary.json", label="scientific parity")
    runtime = _load_digested(activation_dir / "runtime_fingerprint.json", label="runtime fingerprint")
    manifest = _package_manifest()
    expected_cells = _cell_rows(manifest)
    _planning_plan_path, planning_bound_plan = _validate_binding(
        planning.get("execution_plan"), label="planning execution plan"
    )
    _planning_package_path, planning_bound_package = _validate_binding(
        planning.get("package_manifest"), label="planning package manifest"
    )
    _checkpoint_parity_path, checkpoint_parity = _validate_binding(
        parity.get("checkpoint_overlay_parity_canary"),
        label="checkpoint overlay parity canary",
    )
    expected_ra_hashes = {
        row["regime"]: row["protocol_sha256"]
        for row in expected_cells
        if row["method"] == "ra_singleton_plateau"
    }
    if (
        planning.get("schema") != PLANNING_SCHEMA
        or planning.get("status") != "passed_inert_planning"
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "passed_strongest_pairs_first_serial_plan"
        or plan.get("package_manifest_sha256") != manifest["sha256"]
        or plan.get("cells") != expected_cells
        or plan.get("execution_ids")
        != [row["execution_id"] for row in expected_cells]
        or planning_bound_plan != plan
        or planning_bound_package != manifest
        or planning.get("runner") != _binding(RUNNER_PATH)
        or planning.get("worker") != _binding(WORKER_PATH)
        or planning.get("archive_module") != _binding(ARCHIVE_MODULE_PATH)
        or planning.get("maximum_concurrency") != 1
        or planning.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or planning.get("fresh_start_only") is not True
        or planning.get("checkpoint_usage") != CHECKPOINT_USAGE
        or planning.get("checkpoint_resume_authorized") is not False
        or planning.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or plan.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or plan.get("fresh_start_only") is not True
        or plan.get("checkpoint_usage") != CHECKPOINT_USAGE
        or plan.get("checkpoint_resume_authorized") is not False
        or plan.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or planning.get("execution_authorized") is not False
        or planning.get("submission_authorized") is not False
        or plan.get("submission_authorized") is not False
        or planning.get("paper_adoption_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or planning.get("paper_evidence_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("status")
        != "authorized_local_matched_singleton12_execution"
        or activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status") != "authorized_local_execution"
        or parity.get("schema") != PARITY_SCHEMA
        or parity.get("status") != PARITY_STATUS
        or parity.get("package_manifest_sha256") != manifest["sha256"]
        or parity.get("sealed_ra_protocol_sha256_by_regime")
        != expected_ra_hashes
        or parity.get("checkpoint_overlay_parity_scope") != PARITY_CANARY_SCOPE
        or parity.get("multi_round_compact_tail_resume_validated") is not False
        or parity.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or parity.get("fresh_start_only") is not True
        or parity.get("checkpoint_usage") != CHECKPOINT_USAGE
        or parity.get("checkpoint_resume_authorized") is not False
        or parity.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or parity.get("paper_adoption_authorized") is not False
        or parity.get("paper_evidence_adoption_authorized") is not False
        or checkpoint_parity is None
        or _checkpoint_parity_path
        != REPAIR_ROOT
        / (
            "paper_i_page12_strong_holstein_sector5_local_repair_"
            "20260814_v1_activation/scientific_parity_canary.json"
        )
        or checkpoint_parity.get("sha256")
        != "ad870ca15fd75b31400986c71245a56283532a5b5714b1c456185ce87ad0ceaa"
        or runtime.get("schema") != RUNTIME_FINGERPRINT_SCHEMA
        or runtime != _live_runtime_fingerprint()
        or authorization.get("planning_manifest_sha256") != planning["sha256"]
        or authorization.get("execution_plan_sha256") != plan["sha256"]
        or authorization.get("package_manifest_sha256") != manifest["sha256"]
        or authorization.get("runtime_fingerprint_sha256") != runtime["sha256"]
        or authorization.get("scientific_parity_canary_sha256") != parity["sha256"]
        or authorization.get("execution_ids") != plan.get("execution_ids")
        or authorization.get("authorized_by")
        != "user_confirmed_local_paper_i_singleton_rerun"
        or authorization.get("execution_source_policy")
        != EXECUTION_SOURCE_POLICY
        or authorization.get("fresh_start_only") is not True
        or authorization.get("checkpoint_usage") != CHECKPOINT_USAGE
        or authorization.get("checkpoint_resume_authorized") is not False
        or authorization.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or activation.get("package_manifest_sha256") != manifest["sha256"]
        or activation.get("planning_manifest_sha256") != planning["sha256"]
        or activation.get("execution_plan_sha256") != plan["sha256"]
        or activation.get("execution_authorization_sha256")
        != authorization["sha256"]
        or activation.get("scientific_parity_canary_sha256") != parity["sha256"]
        or activation.get("runtime_fingerprint_sha256") != runtime["sha256"]
        or activation.get("runner") != _binding(RUNNER_PATH)
        or activation.get("worker") != _binding(WORKER_PATH)
        or activation.get("archive_module") != _binding(ARCHIVE_MODULE_PATH)
        or activation.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or activation.get("fresh_start_only") is not True
        or activation.get("checkpoint_usage") != CHECKPOINT_USAGE
        or activation.get("checkpoint_resume_authorized") is not False
        or activation.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or activation.get("execution_authorized") is not True
        or activation.get("archive_rotation_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("paper_adoption_authorized") is not False
        or activation.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("execution_authorized") is not True
        or authorization.get("archive_rotation_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
        or activation.get("execution_ids") != plan.get("execution_ids")
    ):
        raise MatchedSingleton12Error("Activation authority drifted.")
    return activation, plan, authorization, runtime


def _validate_handoff(activation: Mapping[str, Any]) -> dict[str, Any]:
    receipt_text = os.environ.get(HANDOFF_RECEIPT_ENV)
    token = os.environ.get(HANDOFF_TOKEN_ENV)
    descriptor_text = os.environ.get(HANDOFF_LOCK_FD_ENV)
    if not receipt_text or not token or not descriptor_text:
        raise MatchedSingleton12Error("Receipt-gated handoff environment is absent.")
    receipt_path = Path(receipt_text)
    if receipt_path.resolve() != DEFAULT_HANDOFF_RECEIPT.resolve():
        raise MatchedSingleton12Error("Handoff receipt path is not the pinned gate path.")
    receipt = _load_digested(receipt_path, label="handoff receipt")
    target_contract = _load_digested(
        DEFAULT_TARGET_CONTRACT, label="matched target contract"
    )
    target = target_contract.get("target")
    if not isinstance(target, Mapping):
        raise MatchedSingleton12Error("Matched target contract omitted target authority.")
    parity = _load_digested(
        DEFAULT_ACTIVATION_DIR / "scientific_parity_canary.json",
        label="matched scientific parity",
    )
    runtime_fingerprint = _load_digested(
        DEFAULT_ACTIVATION_DIR / "runtime_fingerprint.json",
        label="matched runtime fingerprint",
    )
    expected_token = hashlib.sha256(
        f"{receipt['sha256']}:matched-singleton12-target-launch-v1".encode("utf-8")
    ).hexdigest()
    try:
        descriptor = int(descriptor_text)
        descriptor_stat = os.fstat(descriptor)
        lock_stat = DEFAULT_HANDOFF_LOCK.stat(follow_symlinks=False)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError) as exc:
        raise MatchedSingleton12Error("Inherited handoff lock descriptor is invalid.") from exc
    expected_command = [
        PYTHON_EXECUTABLE.as_posix(),
        "-B",
        RUNNER_PATH.as_posix(),
        "--run-campaign",
    ]
    source_runtime = REPO_ROOT / (
        "output/local_runs/"
        "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1"
    )
    source_terminal_path, source_terminal = _validate_binding(
        receipt.get("source_terminal"),
        label="handoff source terminal",
        root=source_runtime,
    )
    source_status_path, source_status = _validate_binding(
        receipt.get("source_final_status"),
        label="handoff source final status",
        root=source_runtime,
    )
    if (
        token != expected_token
        or DEFAULT_HANDOFF_LOCK.is_symlink()
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (lock_stat.st_dev, lock_stat.st_ino)
        or receipt.get("schema") != HANDOFF_RECEIPT_SCHEMA
        or receipt.get("status") != HANDOFF_RECEIPT_STATUS
        or target_contract.get("schema") != TARGET_CONTRACT_SCHEMA
        or target_contract.get("status") != TARGET_CONTRACT_STATUS
        or target_contract.get("gate_script_path") != GATE_PATH.as_posix()
        or target_contract.get("gate_script_sha256") != _sha256_file(GATE_PATH)
        or receipt.get("gate_script") != _binding(GATE_PATH)
        or receipt.get("target_contract")
        != _binding(DEFAULT_TARGET_CONTRACT, canonical=True)
        or receipt.get("target_contract_sha256") != target_contract["sha256"]
        or receipt.get("target_authority_bindings")
        != target.get("authority_bindings")
        or receipt.get("target_activation_manifest_sha256") != activation.get("sha256")
        or receipt.get("target_scientific_parity_canary_sha256")
        != parity["sha256"]
        or receipt.get("live_runtime_fingerprint_sha256")
        != runtime_fingerprint["sha256"]
        or receipt.get("target_command") != expected_command
        or receipt.get("target_command_sha256")
        != hashlib.sha256(_canonical_json_bytes(expected_command)).hexdigest()
        or receipt.get("target_environment") != REQUIRED_NUMERICAL_ENVIRONMENT
        or receipt.get("target_environment_sha256")
        != hashlib.sha256(
            _canonical_json_bytes(REQUIRED_NUMERICAL_ENVIRONMENT)
        ).hexdigest()
        or source_terminal_path != source_runtime / "terminal_receipt.json"
        or source_status_path != source_runtime / "status/campaign.json"
        or source_terminal is None
        or source_terminal.get("status")
        != "passed_all_five_cells_immutable_closure"
        or source_terminal.get("sha256")
        != receipt.get("source_terminal_sha256")
        or source_status is None
        or source_status.get("status") != "passed_all_five_cells"
        or source_status.get("sha256") != receipt.get("source_final_status_sha256")
        or receipt.get("execution_authorized") is not True
        or receipt.get("submission_authorized") is not False
        or receipt.get("paper_adoption_authorized") is not False
        or receipt.get("paper_evidence_adoption_authorized") is not False
        or receipt.get("scientific_execution_performed") is not False
    ):
        raise MatchedSingleton12Error("Receipt-gated handoff authority drifted.")
    return receipt


def _runtime_manifest(activation: Mapping[str, Any], handoff: Mapping[str, Any]) -> dict[str, Any]:
    return _digested(
        {
            "schema": RUNTIME_SCHEMA,
            "status": "authorized_runtime",
            "created_at_utc": _utc_now(),
            "runtime_dir": DEFAULT_RUNTIME_DIR.resolve().as_posix(),
            "activation_manifest_sha256": activation["sha256"],
            "handoff_receipt_sha256": handoff["sha256"],
            "runner_sha256": _sha256_file(RUNNER_PATH),
            "maximum_concurrency": 1,
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "execution_authorized": True,
            "archive_rotation_authorized": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _ensure_runtime(activation: Mapping[str, Any], handoff: Mapping[str, Any]) -> dict[str, Any]:
    path = DEFAULT_RUNTIME_DIR / "runtime_manifest.json"
    if DEFAULT_RUNTIME_DIR.exists() or DEFAULT_RUNTIME_DIR.is_symlink():
        if DEFAULT_RUNTIME_DIR.is_symlink() or not DEFAULT_RUNTIME_DIR.is_dir():
            raise MatchedSingleton12Error("Existing runtime root is unsafe.")
        runtime = _load_digested(path, label="runtime manifest")
        expected = _runtime_manifest(activation, handoff)
        # Creation time is the one immutable observation allowed to differ.
        expected = _digested({**{k: v for k, v in expected.items() if k != "sha256"}, "created_at_utc": runtime.get("created_at_utc")})
        if runtime != expected:
            raise MatchedSingleton12Error("Existing runtime authority drifted.")
        return runtime
    staging = DEFAULT_RUNTIME_DIR.with_name(
        f".{DEFAULT_RUNTIME_DIR.name}.in_progress"
    )
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise MatchedSingleton12Error("Runtime staging root is unsafe.")
        quarantine = DEFAULT_RUNTIME_DIR.with_name(
            f".{DEFAULT_RUNTIME_DIR.name}.quarantine.{time.time_ns()}"
        )
        os.rename(staging, quarantine)
        _fsync_directory(DEFAULT_RUNTIME_DIR.parent)
    staging.mkdir(parents=True, exist_ok=False)
    for name in (
        "runs",
        "receipts",
        "runtime_checks",
        "guards",
        "pair_parity",
        "status",
    ):
        (staging / name).mkdir()
    runtime = _runtime_manifest(activation, handoff)
    _write_json_exclusive(staging / "runtime_manifest.json", runtime)
    if _load_digested(
        staging / "runtime_manifest.json", label="staged runtime manifest"
    ) != runtime:
        raise MatchedSingleton12Error("Staged runtime authority drifted.")
    _fsync_directory(staging)
    if DEFAULT_RUNTIME_DIR.exists() or DEFAULT_RUNTIME_DIR.is_symlink():
        raise FileExistsError(DEFAULT_RUNTIME_DIR)
    os.rename(staging, DEFAULT_RUNTIME_DIR)
    _fsync_directory(DEFAULT_RUNTIME_DIR.parent)
    published = _load_digested(path, label="published runtime manifest")
    if published != runtime:
        raise MatchedSingleton12Error("Published runtime authority drifted.")
    return published


def _runtime_check(recorded: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    execution_id = str(cell["execution_id"])
    receipt_path = (
        DEFAULT_RUNTIME_DIR / "runtime_checks" / f"{execution_id}.json"
    )
    science_evidence_paths = (
        DEFAULT_RUNTIME_DIR / "runs" / execution_id,
        DEFAULT_RUNTIME_DIR / "receipts" / f"{execution_id}.json",
        DEFAULT_RUNTIME_DIR / "guards" / f"{execution_id}.json",
    )
    if any(path.exists() or path.is_symlink() for path in science_evidence_paths):
        raise MatchedSingleton12Error(
            f"Pre-cell runtime check is already bound to science evidence: {execution_id}."
        )
    previous = None
    if receipt_path.exists() or receipt_path.is_symlink():
        previous = _load_digested(
            receipt_path, label=f"previous runtime check {execution_id}"
        )
        if (
            previous.get("schema") != RUNTIME_CHECK_SCHEMA
            or previous.get("status") != "passed_exact_before_cell"
            or previous.get("execution_id") != execution_id
            or previous.get("runtime_fingerprint_sha256")
            != recorded.get("sha256")
        ):
            raise MatchedSingleton12Error(
                f"Previous pre-cell runtime check drifted: {execution_id}."
            )
    live = _live_runtime_fingerprint()
    if live != recorded:
        raise MatchedSingleton12Error(f"Exact runtime fingerprint drifted before {execution_id}.")
    capacity = _capacity(DEFAULT_RUNTIME_DIR)
    archive_capacity = _archive_module().require_regime_launch_capacity(
        DEFAULT_RUNTIME_DIR,
        regime_id=str(cell["regime"]),
        nph=int(cell["n_ph"]),
    )
    if int(capacity["available_memory_bytes"]) < MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise MatchedSingleton12Error(f"Capacity floor failed before {execution_id}.")
    receipt = _digested(
        {
            "schema": RUNTIME_CHECK_SCHEMA,
            "status": "passed_exact_before_cell",
            "checked_at_utc": _utc_now(),
            "execution_id": execution_id,
            "runtime_fingerprint_sha256": live["sha256"],
            "capacity": capacity,
            "archive_campaign_capacity": archive_capacity,
            "replaces_runtime_check_sha256": (
                None if previous is None else previous["sha256"]
            ),
        }
    )
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _child_token(activation: Mapping[str, Any], handoff: Mapping[str, Any], execution_id: str) -> str:
    return hashlib.sha256(
        f"{activation['sha256']}:{handoff['sha256']}:{execution_id}:matched12-child-v1".encode("utf-8")
    ).hexdigest()


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _gone, alive = psutil.wait_procs([*children, parent], timeout=10)
        for item in alive:
            item.kill()
    except psutil.Error:
        process.send_signal(signal.SIGTERM)


def _run_guarded_child(
    *, cell: Mapping[str, Any], activation: Mapping[str, Any], handoff: Mapping[str, Any]
) -> dict[str, Any]:
    execution = str(cell["execution_id"])
    command = [
        PYTHON_EXECUTABLE.as_posix(),
        "-B",
        RUNNER_PATH.as_posix(),
        "--child-cell",
        execution,
    ]
    environment = dict(os.environ)
    environment.update(REQUIRED_NUMERICAL_ENVIRONMENT)
    environment[CHILD_TOKEN_ENV] = _child_token(activation, handoff, execution)
    try:
        handoff_descriptor = int(os.environ[HANDOFF_LOCK_FD_ENV])
        os.fstat(handoff_descriptor)
    except (KeyError, OSError, ValueError) as exc:
        raise MatchedSingleton12Error("Handoff lock cannot be inherited by child.") from exc
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        pass_fds=(handoff_descriptor,),
    )
    started = time.monotonic()
    deadline = started + int(cell["max_runtime_seconds"])
    peak_rss = 0
    minimum_available = 2**63 - 1
    minimum_free_disk = 2**63 - 1
    stop_reason: str | None = None
    try:
        while process.poll() is None:
            aggregate_rss = 0
            try:
                root = psutil.Process(process.pid)
                aggregate_rss = root.memory_info().rss + sum(
                    child.memory_info().rss for child in root.children(recursive=True)
                )
            except psutil.Error:
                pass
            capacity = _capacity(DEFAULT_RUNTIME_DIR)
            peak_rss = max(peak_rss, aggregate_rss)
            minimum_available = min(
                minimum_available, int(capacity["available_memory_bytes"])
            )
            minimum_free_disk = min(
                minimum_free_disk, int(capacity["free_disk_bytes"])
            )
            if time.monotonic() > deadline:
                stop_reason = "maximum_runtime_seconds_exceeded"
            elif aggregate_rss > RSS_LIMIT_BYTES:
                stop_reason = "rss_limit_exceeded"
            elif int(capacity["available_memory_bytes"]) < MINIMUM_AVAILABLE_MEMORY_BYTES:
                stop_reason = "available_memory_floor_breached"
            elif int(capacity["free_disk_bytes"]) < RUNTIME_FREE_DISK_FLOOR_BYTES:
                stop_reason = "runtime_free_disk_floor_breached"
            if stop_reason is not None:
                _terminate_process_tree(process)
                raise MatchedSingleton12Error(
                    f"Runtime guard stopped {execution}: {stop_reason}."
                )
            time.sleep(GUARD_POLL_SECONDS)
        if process.returncode != 0:
            raise MatchedSingleton12Error(f"Cell child failed ({process.returncode}): {execution}")
        guard = _digested(
            {
                "schema": "paper_i_page12_matched_singleton12_guard_receipt_v1",
                "status": "passed",
                "execution_id": execution,
                "max_runtime_seconds": int(cell["max_runtime_seconds"]),
                "elapsed_seconds": time.monotonic() - started,
                "peak_aggregate_rss_bytes": peak_rss,
                "minimum_available_memory_bytes": minimum_available,
                "minimum_free_disk_bytes": minimum_free_disk,
                "rss_limit_bytes": RSS_LIMIT_BYTES,
                "available_memory_floor_bytes": MINIMUM_AVAILABLE_MEMORY_BYTES,
                "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
                "child_returncode": process.returncode,
            }
        )
        _write_json_exclusive(
            DEFAULT_RUNTIME_DIR / "guards" / f"{execution}.json", guard
        )
        return guard
    finally:
        if process.poll() is None:
            _terminate_process_tree(process)


def _native_runtime_semantics_valid(native: Mapping[str, Any]) -> bool:
    python_identity = native.get("python")
    packages = native.get("packages")
    threadpools = native.get("loaded_threadpools")
    libraries = native.get("loaded_blas_lapack_libraries")
    numpy_config = native.get("numpy_configuration")
    scipy_config = native.get("scipy_configuration")
    cpu = native.get("cpu")
    platform_identity = native.get("platform")
    libc_identity = native.get("libc_identity")
    resource = native.get("resource_contract")
    executable_digest = (
        python_identity.get("executable_sha256")
        if isinstance(python_identity, Mapping)
        else None
    )
    if (
        not isinstance(python_identity, Mapping)
        or not Path(str(python_identity.get("executable", ""))).is_absolute()
        or not Path(
            str(python_identity.get("executable_resolved", ""))
        ).is_absolute()
        or not isinstance(executable_digest, str)
        or len(executable_digest) != 64
        or any(character not in "0123456789abcdef" for character in executable_digest)
        or not python_identity.get("version")
        or not python_identity.get("implementation")
        or not isinstance(packages, Mapping)
        or any(not packages.get(name) for name in ("numpy", "scipy", "qiskit"))
        or not isinstance(threadpools, list)
        or not threadpools
        or not isinstance(libraries, list)
        or not libraries
        or not any("blas" in str(path).lower() for path in libraries)
        or not any("lapack" in str(path).lower() for path in libraries)
        or not isinstance(numpy_config, Mapping)
        or not numpy_config
        or not isinstance(scipy_config, Mapping)
        or not scipy_config
        or not isinstance(cpu, Mapping)
        or not isinstance(cpu.get("logical_count"), int)
        or int(cpu["logical_count"]) <= 0
        or not isinstance(cpu.get("physical_count"), int)
        or int(cpu["physical_count"]) <= 0
        or not isinstance(platform_identity, Mapping)
        or not platform_identity.get("machine")
        or not platform_identity.get("system")
        or not isinstance(libc_identity, Mapping)
        or not isinstance(resource, Mapping)
        or native.get("capture_point")
        != "inside_cell_after_numpy_scipy_qiskit_blas_load_before_scientific_execution_v1"
        or native.get("scientific_execution_performed") is not False
        or native.get("submission_authorized") is not False
        or native.get("paper_adoption_authorized") is not False
        or native.get("paper_evidence_adoption_authorized") is not False
    ):
        return False
    try:
        observed_at = datetime.fromisoformat(
            str(native.get("observed_at_utc", "")).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return False
    if observed_at.tzinfo is None or observed_at.utcoffset() != timezone.utc.utcoffset(observed_at):
        return False
    brand = cpu.get("brand_string")
    hardware = cpu.get("mac_hardware_identity")
    features = cpu.get("numpy_dispatch_features")
    affinity = cpu.get("affinity")
    generic_models = {"", "unknown", "generic", "arm", "arm64", "x86_64"}
    brand_value = (
        str(brand.get("value", "")).strip()
        if isinstance(brand, Mapping) and brand.get("available") is True
        else ""
    )
    hardware_model = (
        str(hardware.get("chip_type", "")).strip()
        if isinstance(hardware, Mapping) and hardware.get("available") is True
        else ""
    )
    if (
        brand_value.lower() in generic_models
        and hardware_model.lower() in generic_models
    ):
        return False
    if (
        not isinstance(features, Mapping)
        or not features
        or not any(isinstance(value, bool) for value in features.values())
        or not isinstance(affinity, Mapping)
        or not isinstance(affinity.get("available"), bool)
    ):
        return False
    if affinity["available"]:
        indices = affinity.get("cpu_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(index, int) or index < 0 for index in indices)
        ):
            return False
    elif affinity.get("cpu_indices") is not None:
        return False
    expected_affinity_count = (
        len(affinity["cpu_indices"]) if affinity["available"] else None
    )
    if resource != {
        "kind": "native_local_cpu_only_serial_v1",
        "job_requested_cpu_count": 4,
        "scheduler_allocation_available": False,
        "scheduler_allocated_cpu_count": None,
        "native_local_host_logical_cpu_count": cpu["logical_count"],
        "process_affinity_available": affinity["available"],
        "process_affinity_cpu_count": expected_affinity_count,
        "numerical_kernel_thread_count": 1,
        "maximum_campaign_concurrency": 1,
        "gpu_requested_count": 0,
        "gpu_execution_authorized": False,
        "gpu_execution_active": False,
    }:
        return False
    for row in threadpools:
        if (
            not isinstance(row, Mapping)
            or row.get("user_api") != "blas"
            or not row.get("internal_api")
            or not row.get("filepath")
            or not row.get("version")
            or row.get("num_threads") != 1
            or not row.get("thread_count_source")
        ):
            return False
    stable = sorted(
        threadpools,
        key=lambda row: (
            str(row.get("user_api", "")),
            str(row.get("internal_api", "")),
            str(row.get("prefix", "")),
            str(row.get("filepath", "")),
            str(row.get("version", "")),
        ),
    )
    platform_libc = libc_identity.get("platform_libc_ver")
    libc_images = libc_identity.get("loaded_images")
    loaded_available = libc_identity.get("loaded_image_evidence_available")
    if (
        not isinstance(platform_libc, Mapping)
        or not isinstance(platform_libc.get("available"), bool)
        or platform_libc.get("available")
        != bool(platform_libc.get("name") or platform_libc.get("version"))
        or not isinstance(libc_images, list)
        or not isinstance(loaded_available, bool)
        or loaded_available != bool(libc_images)
    ):
        return False
    if platform_identity.get("system") == "Darwin" and (
        not loaded_available
        or not libc_identity.get("darwin_libsystem_version")
    ):
        return False
    for row in libc_images:
        if (
            not isinstance(row, Mapping)
            or not row.get("path")
            or not row.get("version")
            or not row.get("version_source")
        ):
            return False
    stable_libc_images = sorted(
        libc_images,
        key=lambda row: (
            str(row.get("path", "")),
            str(row.get("version", "")),
            str(row.get("version_source", "")),
        ),
    )
    return threadpools == stable and libc_images == stable_libc_images


def _native_runtime_from_worker_receipt(
    cell: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    execution = str(cell["execution_id"])
    receipt_path = DEFAULT_RUNTIME_DIR / "receipts" / f"{execution}.json"
    receipt = _load_digested(receipt_path, label=f"worker receipt {execution}")
    native = receipt.get("native_runtime_receipt")
    if not isinstance(native, Mapping):
        raise MatchedSingleton12Error("Worker receipt omitted native runtime evidence.")
    if (
        receipt.get("schema")
        != "paper_i_page12_matched_singleton12_worker_receipt_v1"
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution
        or receipt.get("method") != cell["method"]
        or receipt.get("job_spec_sha256") != cell["job_spec_sha256"]
        or receipt.get("fresh_start_only") is not True
        or receipt.get("checkpoint_usage") != CHECKPOINT_USAGE
        or receipt.get("checkpoint_resume_authorized") is not False
        or receipt.get("submission_authorized") is not False
        or receipt.get("paper_adoption_authorized") is not False
        or receipt.get("paper_evidence_adoption_authorized") is not False
        or native.get("schema")
        != "paper_i_page12_matched_singleton12_native_local_runtime_receipt_v1"
        or native.get("execution_id") != execution
        or native.get("method") != cell["method"]
        or native.get("sha256")
        != _canonical_sha256(
            {key: item for key, item in native.items() if key != "sha256"}
        )
        or native.get("numerical_environment")
        != REQUIRED_NUMERICAL_ENVIRONMENT
        or not _native_runtime_semantics_valid(native)
    ):
        raise MatchedSingleton12Error("Worker/native runtime receipt drifted.")
    return receipt, dict(native), receipt_path


def _archive_backed_native_runtime_binding(
    *,
    execution_id: str,
    member_path: str,
    expected_file_sha256: str,
    expected_size_bytes: int,
    expected_canonical_sha256: str,
) -> dict[str, Any]:
    module = _archive_module()
    paths = module.CellArchivePaths(
        runtime_root=DEFAULT_RUNTIME_DIR, execution_id=execution_id
    )
    manifest = _load_digested(
        paths.archive_manifest_path,
        label=f"archive manifest for {execution_id}",
    )
    rows = manifest.get("payload_files")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("path") == member_path
    ] if isinstance(rows, list) else []
    if (
        manifest.get("execution_id") != execution_id
        or manifest.get("source_member_prefix") != f"runs/{execution_id}"
        or len(matches) != 1
        or matches[0].get("sha256") != expected_file_sha256
        or matches[0].get("size_bytes") != expected_size_bytes
    ):
        raise MatchedSingleton12Error(
            "Archived native-runtime member identity drifted."
        )
    if not paths.archive_closure_path.is_file():
        raise MatchedSingleton12Error(
            "Archived native-runtime closure is absent."
        )
    binding: dict[str, Any] = {
        "storage": "authenticated_archive_member_v1",
        "member_path": member_path,
        "member_sha256": expected_file_sha256,
        "member_size_bytes": expected_size_bytes,
        "member_canonical_sha256": expected_canonical_sha256,
        "archive": _binding(paths.archive_path),
        "archive_manifest": _binding(
            paths.archive_manifest_path, canonical=True
        ),
        "archive_closure": _binding(
            paths.archive_closure_path, canonical=True
        ),
    }
    if paths.rotation_intent_path.is_file():
        binding["rotation_intent"] = _binding(
            paths.rotation_intent_path, canonical=True
        )
    if paths.cleanup_receipt_path.is_file():
        binding["cleanup_receipt"] = _binding(
            paths.cleanup_receipt_path, canonical=True
        )
    return binding


def _persistent_cell_evidence(
    cell: Mapping[str, Any], *, require_pair_parity: bool
) -> dict[str, Any]:
    """Reconstruct metadata from receipts retained outside the rotated tree."""

    execution = str(cell["execution_id"])
    receipt, native, receipt_path = _native_runtime_from_worker_receipt(cell)
    activation = _load_digested(
        DEFAULT_ACTIVATION_DIR / "activation_manifest.json",
        label="persistent cell activation",
    )
    runtime_check_path = (
        DEFAULT_RUNTIME_DIR / "runtime_checks" / f"{execution}.json"
    )
    guard_path = DEFAULT_RUNTIME_DIR / "guards" / f"{execution}.json"
    runtime_check = _load_digested(
        runtime_check_path, label=f"runtime check {execution}"
    )
    guard = _load_digested(guard_path, label=f"guard receipt {execution}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise MatchedSingleton12Error("Persistent worker artifact closure is absent.")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise MatchedSingleton12Error("Persistent artifact row is malformed.")
        relative = Path(str(row.get("path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or int(row.get("size_bytes", -1)) < 0
            or len(str(row.get("sha256", ""))) != 64
        ):
            raise MatchedSingleton12Error("Persistent artifact row is unsafe.")
        rows[relative.as_posix()] = row
    required = {
        f"runs/{execution}/execution_manifest.json",
        f"runs/{execution}/checkpoints/current.json",
        f"runs/{execution}/result/estimator_ledger.json",
        f"runs/{execution}/result/result.json",
        f"runs/{execution}/summary/summary.json",
        f"runs/{execution}/runtime/native_runtime.json",
    }
    native_relative = f"runs/{execution}/runtime/native_runtime.json"
    native_row = rows.get(native_relative)
    if (
        not required.issubset(rows)
        or native_row is None
        or native_row.get("sha256")
        != receipt.get("native_runtime_receipt_file_sha256")
        or receipt.get("activation_sha256") != activation["sha256"]
        or runtime_check.get("schema") != RUNTIME_CHECK_SCHEMA
        or runtime_check.get("status") != "passed_exact_before_cell"
        or runtime_check.get("execution_id") != execution
        or guard.get("status") != "passed"
        or guard.get("execution_id") != execution
        or guard.get("max_runtime_seconds") != cell["max_runtime_seconds"]
    ):
        raise MatchedSingleton12Error("Persistent authenticated evidence drifted.")
    native_binding = {
        "path": (DEFAULT_RUNTIME_DIR / native_relative).resolve().as_posix(),
        "size_bytes": int(native_row["size_bytes"]),
        "file_sha256": str(native_row["sha256"]),
        "canonical_sha256": native["sha256"],
    }
    native_archive_binding = _archive_backed_native_runtime_binding(
        execution_id=execution,
        member_path=native_relative,
        expected_file_sha256=str(native_row["sha256"]),
        expected_size_bytes=int(native_row["size_bytes"]),
        expected_canonical_sha256=str(native["sha256"]),
    )
    external_members: dict[str, Path] = {
        "evidence/worker_receipt.json": receipt_path,
        "evidence/supervisor_runtime_check.json": runtime_check_path,
        "evidence/supervisor_guard_receipt.json": guard_path,
    }
    pair = None
    pair_binding = None
    if require_pair_parity:
        pair_path = (
            DEFAULT_RUNTIME_DIR / "pair_parity" / f"{cell['regime']}.json"
        )
        pair = _load_digested(pair_path, label=f"pair parity {cell['regime']}")
        ra_cell = _paired_ra_cell(cell)
        _ra_worker, ra_native, _ra_path = _native_runtime_from_worker_receipt(
            ra_cell
        )
        _validate_pair_parity_receipt(
            pair=pair,
            append_cell=cell,
            ra_cell=ra_cell,
            append_native=native,
            ra_native=ra_native,
        )
        pair_binding = _binding(pair_path, canonical=True)
        external_members["evidence/matched_pair_runtime_parity.json"] = pair_path
    return {
        "worker_receipt": receipt,
        "worker_receipt_binding": _binding(receipt_path, canonical=True),
        "native_runtime_receipt": native,
        "native_runtime_binding": native_binding,
        "native_runtime_archive_binding": native_archive_binding,
        "runtime_check_binding": _binding(runtime_check_path, canonical=True),
        "guard_binding": _binding(guard_path, canonical=True),
        "pair_parity": pair,
        "pair_parity_binding": pair_binding,
        "external_members": external_members,
    }


def _validated_cell_evidence(
    cell: Mapping[str, Any], *, require_pair_parity: bool
) -> dict[str, Any]:
    execution = str(cell["execution_id"])
    receipt, native, receipt_path = _native_runtime_from_worker_receipt(cell)
    runtime_check_path = (
        DEFAULT_RUNTIME_DIR / "runtime_checks" / f"{execution}.json"
    )
    guard_path = DEFAULT_RUNTIME_DIR / "guards" / f"{execution}.json"
    runtime_check = _load_digested(
        runtime_check_path, label=f"runtime check {execution}"
    )
    guard = _load_digested(guard_path, label=f"guard receipt {execution}")
    native_path = DEFAULT_RUNTIME_DIR / "runs" / execution / "runtime/native_runtime.json"
    native_file = _load_digested(
        native_path, label=f"native runtime receipt {execution}"
    )
    execution_manifest_path = (
        DEFAULT_RUNTIME_DIR / "runs" / execution / "execution_manifest.json"
    )
    execution_manifest = _load_digested(
        execution_manifest_path, label=f"cell execution manifest {execution}"
    )
    if (
        receipt.get("activation_sha256")
        != _load_digested(
            DEFAULT_ACTIVATION_DIR / "activation_manifest.json",
            label="cell activation",
        )["sha256"]
        or dict(native) != native_file
        or receipt.get("native_runtime_receipt_file_sha256")
        != _sha256_file(native_path)
        or native.get("numerical_environment")
        != REQUIRED_NUMERICAL_ENVIRONMENT
        or native.get("capture_point")
        != "inside_cell_after_numpy_scipy_qiskit_blas_load_before_scientific_execution_v1"
        or execution_manifest.get("execution_id") != execution
        or execution_manifest.get("activation_sha256")
        != receipt.get("activation_sha256")
        or execution_manifest.get("native_runtime_receipt", {}).get("sha256")
        != native["sha256"]
        or execution_manifest.get("compact_checkpoint_keep_history_tail") != 1
        or receipt.get("execution_manifest_sha256")
        != execution_manifest["sha256"]
        or runtime_check.get("schema") != RUNTIME_CHECK_SCHEMA
        or runtime_check.get("status") != "passed_exact_before_cell"
        or runtime_check.get("execution_id") != execution
        or guard.get("status") != "passed"
        or guard.get("execution_id") != execution
        or guard.get("max_runtime_seconds") != cell["max_runtime_seconds"]
    ):
        raise MatchedSingleton12Error(f"Authenticated cell evidence drifted: {execution}")
    artifact_paths: set[str] = set()
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise MatchedSingleton12Error("Worker artifact closure is absent.")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise MatchedSingleton12Error("Worker artifact row is malformed.")
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise MatchedSingleton12Error("Worker artifact path is unsafe.")
        path = DEFAULT_RUNTIME_DIR / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(path) != row.get("sha256")
        ):
            raise MatchedSingleton12Error(f"Worker artifact drifted: {relative}")
        artifact_paths.add(relative.as_posix())
    required_artifacts = {
        f"runs/{execution}/execution_manifest.json",
        f"runs/{execution}/checkpoints/current.json",
        f"runs/{execution}/result/estimator_ledger.json",
        f"runs/{execution}/result/result.json",
        f"runs/{execution}/summary/summary.json",
        f"runs/{execution}/runtime/native_runtime.json",
    }
    if not required_artifacts.issubset(artifact_paths):
        raise MatchedSingleton12Error("Worker artifact closure is incomplete.")
    external_members: dict[str, Path] = {
        "evidence/worker_receipt.json": receipt_path,
        "evidence/supervisor_runtime_check.json": runtime_check_path,
        "evidence/supervisor_guard_receipt.json": guard_path,
    }
    pair = None
    pair_path = DEFAULT_RUNTIME_DIR / "pair_parity" / f"{cell['regime']}.json"
    if require_pair_parity:
        pair = _load_digested(pair_path, label=f"pair parity {cell['regime']}")
        ra_cell = _paired_ra_cell(cell)
        _ra_worker, ra_native, _ra_path = _native_runtime_from_worker_receipt(
            ra_cell
        )
        _validate_pair_parity_receipt(
            pair=pair,
            append_cell=cell,
            ra_cell=ra_cell,
            append_native=native,
            ra_native=ra_native,
        )
        external_members["evidence/matched_pair_runtime_parity.json"] = pair_path
    return {
        "worker_receipt": receipt,
        "worker_receipt_binding": _binding(receipt_path, canonical=True),
        "native_runtime_receipt": dict(native),
        "native_runtime_binding": _binding(native_path, canonical=True),
        "native_runtime_archive_binding": None,
        "runtime_check_binding": _binding(runtime_check_path, canonical=True),
        "guard_binding": _binding(guard_path, canonical=True),
        "pair_parity": pair,
        "pair_parity_binding": (
            None if pair is None else _binding(pair_path, canonical=True)
        ),
        "external_members": external_members,
    }


def _native_runtime_projection(receipt: Mapping[str, Any]) -> dict[str, Any]:
    allowed_differences = {"sha256", "observed_at_utc", "execution_id", "method"}
    return {
        key: item for key, item in receipt.items() if key not in allowed_differences
    }


def _paired_ra_cell(append_cell: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        cell
        for cell in _cell_rows(_package_manifest())
        if cell.get("method") == "ra_singleton_plateau"
        and cell.get("regime") == append_cell.get("regime")
        and cell.get("n_ph") == append_cell.get("n_ph")
    ]
    if append_cell.get("method") != "append_singleton" or len(matches) != 1:
        raise MatchedSingleton12Error("Matched pair cell identity drifted.")
    return matches[0]


def _validate_pair_parity_receipt(
    *,
    pair: Mapping[str, Any],
    append_cell: Mapping[str, Any],
    ra_cell: Mapping[str, Any],
    append_native: Mapping[str, Any],
    ra_native: Mapping[str, Any],
) -> None:
    ra_projection = _native_runtime_projection(ra_native)
    append_projection = _native_runtime_projection(append_native)
    try:
        created_at = datetime.fromisoformat(
            str(pair.get("created_at_utc", "")).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        created_at = None
    if (
        pair.get("sha256")
        != _canonical_sha256(
            {key: value for key, value in pair.items() if key != "sha256"}
        )
        or pair.get("schema") != PAIR_PARITY_SCHEMA
        or pair.get("status") != "passed_exact_native_runtime_pair_parity"
        or pair.get("regime") != append_cell.get("regime")
        or pair.get("n_ph") != append_cell.get("n_ph")
        or ra_cell.get("regime") != append_cell.get("regime")
        or ra_cell.get("n_ph") != append_cell.get("n_ph")
        or pair.get("ra_execution_id") != ra_cell.get("execution_id")
        or pair.get("append_execution_id") != append_cell.get("execution_id")
        or pair.get("ra_native_runtime_receipt_sha256")
        != ra_native.get("sha256")
        or pair.get("append_native_runtime_receipt_sha256")
        != append_native.get("sha256")
        or ra_projection != append_projection
        or pair.get("science_relevant_projection_sha256")
        != _canonical_sha256(ra_projection)
        or pair.get("allowed_differences")
        != ["execution_id", "method", "observed_at_utc", "sha256"]
        or pair.get("exact_projection_match") is not True
        or created_at is None
        or created_at.tzinfo is None
        or created_at.utcoffset() != timezone.utc.utcoffset(created_at)
        or pair.get("submission_authorized") is not False
        or pair.get("paper_adoption_authorized") is not False
        or pair.get("paper_evidence_adoption_authorized") is not False
    ):
        raise MatchedSingleton12Error("Matched pair runtime receipt drifted.")


def _publish_pair_parity(
    *, append_cell: Mapping[str, Any], ra_cell: Mapping[str, Any]
) -> dict[str, Any]:
    _ra_receipt, ra_native, _ra_path = _native_runtime_from_worker_receipt(
        ra_cell
    )
    _append_receipt, append_native, _append_path = (
        _native_runtime_from_worker_receipt(append_cell)
    )
    ra_projection = _native_runtime_projection(ra_native)
    append_projection = _native_runtime_projection(append_native)
    if ra_projection != append_projection:
        raise MatchedSingleton12Error(
            f"Native runtime parity failed for {append_cell['regime']}."
        )
    receipt = _digested(
        {
            "schema": PAIR_PARITY_SCHEMA,
            "status": "passed_exact_native_runtime_pair_parity",
            "created_at_utc": _utc_now(),
            "regime": append_cell["regime"],
            "n_ph": append_cell["n_ph"],
            "ra_execution_id": ra_cell["execution_id"],
            "append_execution_id": append_cell["execution_id"],
            "ra_native_runtime_receipt_sha256": ra_native["sha256"],
            "append_native_runtime_receipt_sha256": append_native["sha256"],
            "science_relevant_projection_sha256": _canonical_sha256(
                ra_projection
            ),
            "allowed_differences": [
                "execution_id",
                "method",
                "observed_at_utc",
                "sha256",
            ],
            "exact_projection_match": True,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    path = DEFAULT_RUNTIME_DIR / "pair_parity" / f"{append_cell['regime']}.json"
    _write_json_exclusive(path, receipt)
    return receipt


def _archive_metadata(
    *,
    cell: Mapping[str, Any],
    activation: Mapping[str, Any],
    handoff: Mapping[str, Any],
    runtime: Mapping[str, Any],
    direct_evidence: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    loader = _validated_cell_evidence if direct_evidence else _persistent_cell_evidence
    evidence = loader(
        cell, require_pair_parity=cell["method"] == "append_singleton"
    )
    authority = {
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "activation_manifest_sha256": activation["sha256"],
        "handoff_receipt_sha256": handoff["sha256"],
        "runtime_manifest_sha256": runtime["sha256"],
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }
    metadata = {
        "execution_id": cell["execution_id"],
        "method": cell["method"],
        "regime": cell["regime"],
        "n_ph": cell["n_ph"],
        "job_spec_sha256": cell["job_spec_sha256"],
        "protocol_sha256": cell["protocol_sha256"],
        "route_contract_sha256": cell["route_contract_sha256"],
        "compact_checkpoint_keep_history_tail": 1,
        "execution_source_policy": EXECUTION_SOURCE_POLICY,
        "fresh_start_only": True,
        "checkpoint_usage": CHECKPOINT_USAGE,
        "checkpoint_resume_authorized": False,
        "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
        "worker_receipt": evidence["worker_receipt_binding"],
        "native_runtime_historical_source_member": evidence[
            "native_runtime_binding"
        ],
        "supervisor_runtime_check": evidence["runtime_check_binding"],
        "supervisor_guard_receipt": evidence["guard_binding"],
        "matched_pair_runtime_parity": evidence["pair_parity_binding"],
    }
    return authority, metadata, evidence


def _drive_archive_rotation_state(
    *,
    module: Any,
    paths: Any,
    state: Mapping[str, Any],
    source_member_prefix: str,
    external_members: Mapping[str, Path],
    authority_metadata: Mapping[str, Any],
    cell_metadata: Mapping[str, Any],
    rotation_authority: Mapping[str, Any],
    limits: Any,
) -> dict[str, Any]:
    """Resume exactly the remaining authenticated archive stages."""

    source_pipeline_states = {
        "direct_unarchived",
        "archive_published_pending_manifest",
        "manifest_published_pending_closure",
        "closure_published_pending_intent",
        "intent_published_pending_rename",
    }
    temporary_discard_states = {
        "direct_unarchived",
        "archive_published_pending_manifest",
        "manifest_published_pending_closure",
    }
    rotation_only_states = {
        "retiring_pending_removal",
        "cleanup_receipt_pending",
    }
    state_name = str(state.get("state", ""))
    stale = state.get("stale_archive_temporaries")
    if not isinstance(stale, list):
        raise MatchedSingleton12Error("Archive temporary-state receipt is malformed.")
    if stale:
        if (
            state_name not in temporary_discard_states
            or state.get("source_present") is not True
        ):
            raise MatchedSingleton12Error(
                "Stale archive temporaries are unsafe in this rotation state."
            )
        removed = module.discard_stale_archive_temporaries(paths)
        if sorted(removed) != sorted(str(item) for item in stale):
            raise MatchedSingleton12Error(
                "Stale archive temporary cleanup receipt drifted."
            )
        state = module.inspect_rotation_state(paths)
        state_name = str(state.get("state", ""))
        if state.get("stale_archive_temporaries") != []:
            raise MatchedSingleton12Error(
                "Archive temporaries remained after module-owned cleanup."
            )

    if state_name in source_pipeline_states:
        module.build_cell_archive(
            paths=paths,
            source_member_prefix=source_member_prefix,
            external_members=external_members,
            authority_metadata=authority_metadata,
            cell_metadata=cell_metadata,
            limits=limits,
        )
        module.publish_archive_closure(
            paths=paths,
            source_member_prefix=source_member_prefix,
            authority_metadata=authority_metadata,
            cell_metadata=cell_metadata,
            limits=limits,
        )
        module.publish_rotation_intent(
            paths=paths,
            source_member_prefix=source_member_prefix,
            authority_metadata=authority_metadata,
            cell_metadata=cell_metadata,
            rotation_authority=rotation_authority,
            limits=limits,
        )
        module.complete_safe_tree_rotation(
            paths=paths,
            source_member_prefix=source_member_prefix,
            authority_metadata=authority_metadata,
            cell_metadata=cell_metadata,
            rotation_authority=rotation_authority,
            limits=limits,
        )
    elif state_name in rotation_only_states:
        module.complete_safe_tree_rotation(
            paths=paths,
            source_member_prefix=source_member_prefix,
            authority_metadata=authority_metadata,
            cell_metadata=cell_metadata,
            rotation_authority=rotation_authority,
            limits=limits,
        )
    elif state_name != "archived_closed":
        raise MatchedSingleton12Error(
            f"Unsupported archive restart state: {state_name or 'missing'}"
        )

    return module.validate_archive_backed_closure(
        paths=paths,
        source_member_prefix=source_member_prefix,
        expected_authority_metadata=authority_metadata,
        expected_cell_metadata=cell_metadata,
        limits=limits,
        expected_rotation_authority=rotation_authority,
        require_cleanup=True,
    )


def _archive_and_rotate(
    *,
    module: Any,
    cell: Mapping[str, Any],
    activation: Mapping[str, Any],
    authorization: Mapping[str, Any],
    handoff: Mapping[str, Any],
    runtime: Mapping[str, Any],
    direct_evidence: bool = True,
) -> dict[str, Any]:
    execution = str(cell["execution_id"])
    paths = module.CellArchivePaths(runtime_root=DEFAULT_RUNTIME_DIR, execution_id=execution)
    limits = _archive_limits(module)
    authority, metadata, evidence = _archive_metadata(
        cell=cell,
        activation=activation,
        handoff=handoff,
        runtime=runtime,
        direct_evidence=direct_evidence,
    )
    prefix = f"runs/{execution}"
    state = module.inspect_rotation_state(paths)
    return _drive_archive_rotation_state(
        module=module,
        paths=paths,
        source_member_prefix=prefix,
        external_members=evidence["external_members"],
        authority_metadata=authority,
        cell_metadata=metadata,
        rotation_authority=authorization,
        limits=limits,
        state=state,
    )


def _existing_archive_closure(
    *, module: Any, cell: Mapping[str, Any], activation: Mapping[str, Any], authorization: Mapping[str, Any], handoff: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any] | None:
    execution = str(cell["execution_id"])
    paths = module.CellArchivePaths(runtime_root=DEFAULT_RUNTIME_DIR, execution_id=execution)
    state = module.inspect_rotation_state(paths)
    if state.get("state") == "empty":
        return None
    return _archive_and_rotate(
        module=module,
        cell=cell,
        activation=activation,
        authorization=authorization,
        handoff=handoff,
        runtime=runtime,
        direct_evidence=bool(state.get("source_present")),
    )


def _status(status: str, **fields: Any) -> dict[str, Any]:
    return _digested(
        {
            "schema": STATUS_SCHEMA,
            "status": status,
            "updated_at_utc": _utc_now(),
            **fields,
        }
    )


def _terminal_status_receipt(
    *,
    terminal: Mapping[str, Any],
    execution_ids: Sequence[str],
    updated_at_utc: str,
) -> dict[str, Any]:
    return _digested(
        {
            "schema": STATUS_SCHEMA,
            "status": "passed_all_twelve_cells",
            "updated_at_utc": updated_at_utc,
            "current_execution_id": None,
            "completed_execution_ids": list(execution_ids),
            "terminal_completed_at_utc": terminal.get("completed_at_utc"),
            "terminal_receipt_sha256": terminal.get("sha256"),
        }
    )


def _load_terminal_publication_promise(
    execution_ids: Sequence[str],
) -> dict[str, Any] | None:
    terminal_path = DEFAULT_RUNTIME_DIR / "terminal_receipt.json"
    status_path = DEFAULT_RUNTIME_DIR / "status/campaign.json"
    if terminal_path.exists() or terminal_path.is_symlink():
        return None
    if not status_path.exists() and not status_path.is_symlink():
        return None
    status = _load_digested(status_path, label="terminal publication promise")
    if status.get("status") != "passed_all_twelve_cells":
        return None
    try:
        completed_at = datetime.fromisoformat(
            str(status.get("terminal_completed_at_utc", "")).replace(
                "Z", "+00:00"
            )
        )
        updated_at = datetime.fromisoformat(
            str(status.get("updated_at_utc", "")).replace("Z", "+00:00")
        )
    except (TypeError, ValueError) as exc:
        raise MatchedSingleton12Error(
            "Terminal publication promise timestamp drifted."
        ) from exc
    terminal_digest = str(status.get("terminal_receipt_sha256", ""))
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("current_execution_id") is not None
        or status.get("completed_execution_ids") != list(execution_ids)
        or len(terminal_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in terminal_digest
        )
        or completed_at.tzinfo is None
        or completed_at.utcoffset()
        != timezone.utc.utcoffset(completed_at)
        or updated_at.tzinfo is None
        or updated_at.utcoffset() != timezone.utc.utcoffset(updated_at)
    ):
        raise MatchedSingleton12Error(
            "Terminal publication promise drifted."
        )
    return status


def _publish_terminal_last(
    *, terminal: Mapping[str, Any], status: Mapping[str, Any]
) -> None:
    terminal_path = DEFAULT_RUNTIME_DIR / "terminal_receipt.json"
    status_path = DEFAULT_RUNTIME_DIR / "status/campaign.json"
    expected_status = _terminal_status_receipt(
        terminal=terminal,
        execution_ids=[str(item) for item in terminal.get("execution_ids", [])],
        updated_at_utc=str(status.get("updated_at_utc", "")),
    )
    if status != expected_status:
        raise MatchedSingleton12Error(
            "Terminal status promise does not bind the exact terminal."
        )
    if terminal_path.exists() or terminal_path.is_symlink():
        existing_terminal = _load_digested(
            terminal_path, label="existing terminal receipt"
        )
        existing_status = _load_digested(
            status_path, label="existing terminal status"
        )
        if existing_terminal != dict(terminal) or existing_status != dict(status):
            raise MatchedSingleton12Error(
                "Existing terminal publication drifted."
            )
        return
    # The authenticated status is the deterministic publication promise.  It
    # is durable before the terminal becomes visible, so every visible terminal
    # already has its exact required final status.
    _write_json_atomic(status_path, status)
    _write_json_atomic_noreplace(terminal_path, terminal)


def run_campaign() -> dict[str, Any]:
    activation, plan, authorization, recorded_runtime = _validate_activation()
    handoff = _validate_handoff(activation)
    module = _archive_module()
    runtime_preexisting = DEFAULT_RUNTIME_DIR.exists()
    if not runtime_preexisting:
        module.require_campaign_capacity(
            _existing_capacity_probe(DEFAULT_RUNTIME_DIR)
        )
    runtime = _ensure_runtime(activation, handoff)
    lock_path = DEFAULT_RUNTIME_DIR / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MatchedSingleton12Error("Another matched-12 supervisor owns the lock.") from exc
        completed: list[str] = []
        closures: list[dict[str, Any]] = []
        terminal_publication_started = False
        cells_by_pair_method = {
            (str(cell["regime"]), str(cell["method"])): cell
            for cell in plan["cells"]
        }
        try:
            for cell in plan["cells"]:
                execution = str(cell["execution_id"])
                pair_path = (
                    DEFAULT_RUNTIME_DIR
                    / "pair_parity"
                    / f"{cell['regime']}.json"
                )
                if (
                    cell["method"] == "append_singleton"
                    and (DEFAULT_RUNTIME_DIR / "runs" / execution).is_dir()
                    and not pair_path.exists()
                ):
                    _publish_pair_parity(
                        append_cell=cell,
                        ra_cell=cells_by_pair_method[
                            (str(cell["regime"]), "ra_singleton_plateau")
                        ],
                    )
                existing = _existing_archive_closure(
                    module=module,
                    cell=cell,
                    activation=activation,
                    authorization=authorization,
                    handoff=handoff,
                    runtime=runtime,
                )
                if existing is not None:
                    completed.append(str(cell["execution_id"]))
                    closures.append(existing)
                    continue
                if (DEFAULT_RUNTIME_DIR / "runs" / execution).exists():
                    raise MatchedSingleton12Error(
                        f"Unarchived partial cell requires inspection: {execution}"
                    )
                runtime_check = _runtime_check(recorded_runtime, cell)
                _write_json_atomic(
                    DEFAULT_RUNTIME_DIR / "status/campaign.json",
                    _status(
                        "running_serial_cell",
                        current_execution_id=execution,
                        completed_execution_ids=completed,
                        runtime_check_sha256=runtime_check["sha256"],
                    ),
                )
                _run_guarded_child(
                    cell=cell, activation=activation, handoff=handoff
                )
                if cell["method"] == "append_singleton":
                    _publish_pair_parity(
                        append_cell=cell,
                        ra_cell=cells_by_pair_method[
                            (str(cell["regime"]), "ra_singleton_plateau")
                        ],
                    )
                closure = _archive_and_rotate(
                    module=module,
                    cell=cell,
                    activation=activation,
                    authorization=authorization,
                    handoff=handoff,
                    runtime=runtime,
                )
                completed.append(execution)
                closures.append(closure)
                _write_json_atomic(
                    DEFAULT_RUNTIME_DIR / "status/campaign.json",
                    _status(
                        "cell_passed_archived_pending_remaining",
                        current_execution_id=None,
                        completed_execution_ids=completed,
                        archive_closure_sha256=closure.get("sha256"),
                    ),
                )
            terminal_cell_evidence = []
            for cell in plan["cells"]:
                evidence = _persistent_cell_evidence(
                    cell,
                    require_pair_parity=cell["method"] == "append_singleton",
                )
                terminal_cell_evidence.append(
                    {
                        "execution_id": cell["execution_id"],
                        "method": cell["method"],
                        "worker_receipt": evidence["worker_receipt_binding"],
                        "native_runtime_archive_member": evidence[
                            "native_runtime_archive_binding"
                        ],
                        "native_runtime_historical_source_member": evidence[
                            "native_runtime_binding"
                        ],
                        "supervisor_runtime_check": evidence[
                            "runtime_check_binding"
                        ],
                        "supervisor_guard_receipt": evidence["guard_binding"],
                        "matched_pair_runtime_parity": evidence[
                            "pair_parity_binding"
                        ],
                    }
                )
            pair_parity_bindings = [
                _binding(
                    DEFAULT_RUNTIME_DIR / "pair_parity" / f"{regime}.json",
                    canonical=True,
                )
                for regime in (
                    "strong_strong_u8",
                    "intermediate_strong",
                    "weak_strong",
                    "strong_weak_u8",
                    "intermediate_weak",
                    "weak_weak",
                )
            ]
            publication_promise = _load_terminal_publication_promise(
                plan["execution_ids"]
            )
            terminal_completed_at = (
                str(publication_promise["terminal_completed_at_utc"])
                if publication_promise is not None
                else _utc_now()
            )
            terminal = _digested(
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": TERMINAL_STATUS,
                    "completed_at_utc": terminal_completed_at,
                    "execution_ids": plan["execution_ids"],
                    "completed_execution_ids": completed,
                    "activation_manifest_sha256": activation["sha256"],
                    "execution_authorization_sha256": authorization["sha256"],
                    "handoff_receipt_sha256": handoff["sha256"],
                    "runtime_manifest_sha256": runtime["sha256"],
                    "runtime_fingerprint_sha256": recorded_runtime["sha256"],
                    "archive_closures": closures,
                    "cell_evidence": terminal_cell_evidence,
                    "matched_pair_runtime_parity_receipts": (
                        pair_parity_bindings
                    ),
                    "maximum_concurrency": 1,
                    "compact_checkpoint_keep_history_tail": 1,
                    "execution_source_policy": EXECUTION_SOURCE_POLICY,
                    "fresh_start_only": True,
                    "checkpoint_usage": CHECKPOINT_USAGE,
                    "checkpoint_resume_authorized": False,
                    "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
                    "execution_authorized": True,
                    "archive_rotation_authorized": True,
                    "submission_authorized": False,
                    "paper_adoption_authorized": False,
                    "paper_evidence_adoption_authorized": False,
                }
            )
            terminal_status = _terminal_status_receipt(
                terminal=terminal,
                execution_ids=plan["execution_ids"],
                updated_at_utc=(
                    str(publication_promise["updated_at_utc"])
                    if publication_promise is not None
                    else _utc_now()
                ),
            )
            if (
                publication_promise is not None
                and terminal_status != publication_promise
            ):
                raise MatchedSingleton12Error(
                    "Reconstructed terminal did not match its publication promise."
                )
            terminal_publication_started = True
            _publish_terminal_last(terminal=terminal, status=terminal_status)
            return terminal
        except BaseException as exc:
            if not terminal_publication_started:
                _write_json_atomic(
                    DEFAULT_RUNTIME_DIR / "status/campaign.json",
                    _status(
                        "failed_or_guard_stopped",
                        current_execution_id=None,
                        completed_execution_ids=completed,
                        failure={"type": type(exc).__name__, "message": str(exc)},
                    ),
                )
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_completed_terminal_read_only() -> dict[str, Any]:
    """Deeply rescan the immutable matched-12 terminal without mutation."""

    activation, plan, authorization, recorded_runtime = _validate_activation()
    handoff = _load_digested(
        DEFAULT_HANDOFF_RECEIPT, label="completed handoff receipt"
    )
    runtime = _load_digested(
        DEFAULT_RUNTIME_DIR / "runtime_manifest.json",
        label="completed runtime manifest",
    )
    terminal = _load_digested(
        DEFAULT_RUNTIME_DIR / "terminal_receipt.json",
        label="matched-12 terminal receipt",
    )
    status = _load_digested(
        DEFAULT_RUNTIME_DIR / "status/campaign.json",
        label="matched-12 terminal status",
    )
    if (
        handoff.get("schema") != HANDOFF_RECEIPT_SCHEMA
        or handoff.get("status") != HANDOFF_RECEIPT_STATUS
        or handoff.get("target_activation_manifest_sha256")
        != activation["sha256"]
        or handoff.get("execution_authorized") is not True
        or handoff.get("submission_authorized") is not False
        or handoff.get("paper_adoption_authorized") is not False
        or handoff.get("paper_evidence_adoption_authorized") is not False
        or runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("status") != "authorized_runtime"
        or runtime.get("runtime_dir") != DEFAULT_RUNTIME_DIR.resolve().as_posix()
        or runtime.get("activation_manifest_sha256") != activation["sha256"]
        or runtime.get("handoff_receipt_sha256") != handoff["sha256"]
        or runtime.get("runner_sha256") != _sha256_file(RUNNER_PATH)
        or runtime.get("maximum_concurrency") != 1
        or runtime.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or runtime.get("fresh_start_only") is not True
        or runtime.get("checkpoint_usage") != CHECKPOINT_USAGE
        or runtime.get("checkpoint_resume_authorized") is not False
        or runtime.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or runtime.get("execution_authorized") is not True
        or runtime.get("archive_rotation_authorized") is not True
        or runtime.get("submission_authorized") is not False
        or runtime.get("paper_adoption_authorized") is not False
        or runtime.get("paper_evidence_adoption_authorized") is not False
    ):
        raise MatchedSingleton12Error("Completed runtime authority drifted.")

    module = _archive_module()
    limits = _archive_limits(module)
    closures: list[dict[str, Any]] = []
    cell_evidence: list[dict[str, Any]] = []
    cells_by_pair_method = {
        (str(cell["regime"]), str(cell["method"])): cell
        for cell in plan["cells"]
    }
    for cell in plan["cells"]:
        authority, metadata, evidence = _archive_metadata(
            cell=cell,
            activation=activation,
            handoff=handoff,
            runtime=runtime,
            direct_evidence=False,
        )
        execution = str(cell["execution_id"])
        paths = module.CellArchivePaths(
            runtime_root=DEFAULT_RUNTIME_DIR, execution_id=execution
        )
        closure = module.validate_archive_backed_closure(
            paths=paths,
            source_member_prefix=f"runs/{execution}",
            expected_authority_metadata=authority,
            expected_cell_metadata=metadata,
            limits=limits,
            expected_rotation_authority=authorization,
            require_cleanup=True,
        )
        if evidence.get("native_runtime_archive_binding") is None:
            raise MatchedSingleton12Error(
                "Terminal native-runtime archive binding is absent."
            )
        pair = evidence.get("pair_parity")
        if cell["method"] == "append_singleton":
            ra_cell = cells_by_pair_method[
                (str(cell["regime"]), "ra_singleton_plateau")
            ]
            _ra_worker, ra_native, _ra_path = _native_runtime_from_worker_receipt(
                ra_cell
            )
            append_native = evidence["native_runtime_receipt"]
            if not isinstance(pair, Mapping):
                raise MatchedSingleton12Error(
                    "Terminal matched-pair runtime closure drifted."
                )
            _validate_pair_parity_receipt(
                pair=pair,
                append_cell=cell,
                ra_cell=ra_cell,
                append_native=append_native,
                ra_native=ra_native,
            )
        closures.append(closure)
        cell_evidence.append(
            {
                "execution_id": execution,
                "method": cell["method"],
                "worker_receipt": evidence["worker_receipt_binding"],
                "native_runtime_archive_member": evidence[
                    "native_runtime_archive_binding"
                ],
                "native_runtime_historical_source_member": evidence[
                    "native_runtime_binding"
                ],
                "supervisor_runtime_check": evidence[
                    "runtime_check_binding"
                ],
                "supervisor_guard_receipt": evidence["guard_binding"],
                "matched_pair_runtime_parity": evidence[
                    "pair_parity_binding"
                ],
            }
        )
    pair_bindings = [
        _binding(
            DEFAULT_RUNTIME_DIR / "pair_parity" / f"{regime}.json",
            canonical=True,
        )
        for regime in (
            "strong_strong_u8",
            "intermediate_strong",
            "weak_strong",
            "strong_weak_u8",
            "intermediate_weak",
            "weak_weak",
        )
    ]
    execution_ids = [str(cell["execution_id"]) for cell in plan["cells"]]
    if (
        terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("status") != TERMINAL_STATUS
        or terminal.get("execution_ids") != execution_ids
        or terminal.get("completed_execution_ids") != execution_ids
        or terminal.get("activation_manifest_sha256") != activation["sha256"]
        or terminal.get("execution_authorization_sha256")
        != authorization["sha256"]
        or terminal.get("handoff_receipt_sha256") != handoff["sha256"]
        or terminal.get("runtime_manifest_sha256") != runtime["sha256"]
        or terminal.get("runtime_fingerprint_sha256")
        != recorded_runtime["sha256"]
        or terminal.get("archive_closures") != closures
        or terminal.get("cell_evidence") != cell_evidence
        or terminal.get("matched_pair_runtime_parity_receipts")
        != pair_bindings
        or terminal.get("maximum_concurrency") != 1
        or terminal.get("compact_checkpoint_keep_history_tail") != 1
        or terminal.get("execution_source_policy") != EXECUTION_SOURCE_POLICY
        or terminal.get("fresh_start_only") is not True
        or terminal.get("checkpoint_usage") != CHECKPOINT_USAGE
        or terminal.get("checkpoint_resume_authorized") is not False
        or terminal.get("sealed_resume_reader_sha256")
        != SEALED_RESUME_READER_SHA256
        or terminal.get("execution_authorized") is not True
        or terminal.get("archive_rotation_authorized") is not True
        or terminal.get("submission_authorized") is not False
        or terminal.get("paper_adoption_authorized") is not False
        or terminal.get("paper_evidence_adoption_authorized") is not False
        or status.get("schema") != STATUS_SCHEMA
        or status.get("status") != "passed_all_twelve_cells"
        or status.get("completed_execution_ids") != execution_ids
        or status.get("current_execution_id") is not None
        or status.get("terminal_completed_at_utc")
        != terminal.get("completed_at_utc")
        or status.get("terminal_receipt_sha256") != terminal["sha256"]
    ):
        raise MatchedSingleton12Error(
            "Matched-12 deep terminal closure drifted."
        )
    return terminal


def run_child_cell(execution_id: str) -> dict[str, Any]:
    activation, plan, _authorization, recorded_runtime = _validate_activation()
    handoff = _validate_handoff(activation)
    cells = {str(cell["execution_id"]): cell for cell in plan["cells"]}
    cell = cells.get(execution_id)
    if cell is None:
        raise MatchedSingleton12Error("Unknown child execution ID.")
    if os.environ.get(CHILD_TOKEN_ENV) != _child_token(activation, handoff, execution_id):
        raise MatchedSingleton12Error("Child execution token drifted.")
    if _live_runtime_fingerprint() != recorded_runtime:
        raise MatchedSingleton12Error("Child runtime fingerprint drifted.")
    worker = _load_module(WORKER_PATH, "paper_i_page12_matched12_worker_child")
    return worker.run_cell(
        job_path=Path(str(cell["job_path"])),
        activation_path=DEFAULT_ACTIVATION_DIR / "activation_manifest.json",
        output_dir=DEFAULT_RUNTIME_DIR / "runs" / execution_id,
        receipt_path=DEFAULT_RUNTIME_DIR / "receipts" / f"{execution_id}.json",
        child_token=os.environ[CHILD_TOKEN_ENV],
    )


def inert_preflight() -> dict[str, Any]:
    manifest = _package_manifest()
    cells = _cell_rows(manifest)
    activation_state = "absent"
    if DEFAULT_ACTIVATION_DIR.exists():
        _validate_activation()
        activation_state = "validated_preauthorized"
    return _digested(
        {
            "schema": "paper_i_page12_matched_singleton12_local_preflight_v1",
            "status": "passed_inert_preflight",
            "package_manifest_sha256": manifest["sha256"],
            "row_count": len(cells),
            "methods": sorted({str(cell["method"]) for cell in cells}),
            "activation_state": activation_state,
            "runtime_state": "present" if DEFAULT_RUNTIME_DIR.exists() else "absent",
            "capacity": _capacity(DEFAULT_RUNTIME_DIR),
            "execution_source_policy": EXECUTION_SOURCE_POLICY,
            "fresh_start_only": True,
            "checkpoint_usage": CHECKPOINT_USAGE,
            "checkpoint_resume_authorized": False,
            "sealed_resume_reader_sha256": SEALED_RESUME_READER_SHA256,
            "scientific_execution_performed": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-planning", action="store_true")
    modes.add_argument("--authorize-activation", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run-campaign", action="store_true")
    modes.add_argument("--child-cell")
    args = parser.parse_args()
    try:
        if args.prepare_planning:
            payload = prepare_planning()
        elif args.authorize_activation:
            payload = authorize_activation()
        elif args.preflight:
            payload = inert_preflight()
        elif args.run_campaign:
            payload = run_campaign()
        else:
            payload = run_child_cell(str(args.child_cell))
        print(_canonical_json_bytes(payload).decode("utf-8"), flush=True)
        return 0
    except (
        FileExistsError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        psutil.Error,
        MatchedSingleton12Error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
