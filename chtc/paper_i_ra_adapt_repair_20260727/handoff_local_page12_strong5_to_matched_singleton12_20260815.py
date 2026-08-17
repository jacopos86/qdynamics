#!/usr/bin/env python3
"""Receipt-gated one-shot handoff from strong-sector repair to matched 12.

The gate is deliberately target-agnostic until an immutable, self-digested
target contract is materialized beside it.  It never constructs scientific
settings, authorizes execution, calls the remote-runner HTTP API, or adopts
paper evidence.  Once the existing five-cell campaign has immutable closure,
the gate validates the preauthorized target, writes a no-replace handoff
receipt, and replaces itself with the exact target command.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import psutil


GATE_PATH = Path(__file__).resolve()
REPAIR_ROOT = GATE_PATH.parent
REPO_ROOT = GATE_PATH.parents[2]
PYTHON_EXECUTABLE = Path(
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
)

SOURCE_RUNNER = REPAIR_ROOT / (
    "run_local_page12_strong_holstein_sector5_20260814.py"
)
SOURCE_RUNNER_SHA256 = (
    "d0e20540f0217364adc47df2c90a8f594469c70f99d32ad280dfd95c2482d8cb"
)
SOURCE_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1_activation"
)
SOURCE_ACTIVATION_SHA256 = (
    "7b0851d108eeb15e5285df6c3745fa85befe25cec122811a338321a7b9b94518"
)
SOURCE_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1"
)
SOURCE_RUNTIME_SHA256 = (
    "0f7697f0d4cda4d74705339138b668489038df2519e66a5035bfc776f834b031"
)
SOURCE_TERMINAL_SCHEMA = (
    "paper_i_page12_strong_sector5_local_terminal_receipt_v1"
)
SOURCE_TERMINAL_STATUS = "passed_all_five_cells_immutable_closure"
SOURCE_FINAL_STATUS = "passed_all_five_cells"
SOURCE_EXECUTION_IDS = (
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_strong_u8__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_always_commutation_reduced",
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_strong_u8__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_append_only",
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "intermediate_strong__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_always_commutation_reduced",
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "intermediate_strong__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_append_only",
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "weak_strong__nph7__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_always_commutation_reduced",
)

DEFAULT_TARGET_CONTRACT = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_target_contract_20260815.json"
)
DEFAULT_STATE_DIR = REPAIR_ROOT / (
    "paper_i_matched_singleton12_after_strong5_handoff_state_20260815_v1"
)

TARGET_CONTRACT_SCHEMA = "paper_i_matched_singleton12_handoff_contract_v1"
TARGET_CONTRACT_STATUS = "passed_target_campaign_preauthorized"
HANDOFF_RECEIPT_SCHEMA = "paper_i_matched_singleton12_handoff_receipt_v1"
HANDOFF_RECEIPT_STATUS = (
    "passed_source_terminal_and_target_activation_authorized_pending_exec"
)
GATE_STATUS_SCHEMA = "paper_i_matched_singleton12_handoff_gate_status_v1"
PREFLIGHT_SCHEMA = "paper_i_matched_singleton12_handoff_preflight_v1"
SOURCE_STATUS_REPAIR_SCHEMA = (
    "paper_i_matched_singleton12_source_final_status_repair_v1"
)

HANDOFF_RECEIPT_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_RECEIPT"
HANDOFF_TOKEN_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_TOKEN"
HANDOFF_LOCK_FD_ENV = "PAPER_I_MATCHED_SINGLETON12_HANDOFF_LOCK_FD"

POLL_SECONDS = 20.0
WAIT_TIMEOUT_SECONDS = 42 * 24 * 60 * 60
HARD_MINIMUM_FREE_DISK_BYTES = 6 * 1024**3
HARD_MINIMUM_AVAILABLE_MEMORY_BYTES = 2 * 1024**3
ALLOWED_RUNNING_SOURCE_STATUSES = {
    "running_serial_cell",
    "cell_passed_pending_remaining",
}
FAILURE_SOURCE_STATUSES = {"failed_or_guard_stopped"}
TARGET_METHODS = {"ra_singleton_plateau", "append_singleton"}
TARGET_REGIMES = {
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
}
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
BASE_SCIENTIFIC_OVERLAP_MARKERS = (
    ("--run-cell",),
    ("local_runner.py", "run-cell"),
    ("run_cell.py", "--run"),
)


class HandoffGateError(RuntimeError):
    """A fail-closed handoff validation or transition error."""


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HandoffGateError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HandoffGateError(f"{label} must be a JSON object.")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != _canonical_sha256(unsigned):
        raise HandoffGateError(f"{label} self-digest drifted.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise HandoffGateError(f"Stale atomic-write temporary: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic_noreplace(
    path: Path, value: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.publish")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if temporary.exists() or temporary.is_symlink():
        raise HandoffGateError(f"Stale no-replace temporary: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _binding(path: Path, *, canonical: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HandoffGateError(f"Cannot bind absent or unsafe file: {path}")
    value: dict[str, Any] = {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha256_file(path),
    }
    if canonical:
        value["canonical_sha256"] = _load_digested(
            path, label=f"canonical binding {path.name}"
        )["sha256"]
    return value


def _validate_binding(
    binding: Mapping[str, Any], *, label: str, root: Path | None = None
) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(binding, Mapping):
        raise HandoffGateError(f"{label} binding is malformed.")
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute():
        raise HandoffGateError(f"{label} binding path must be absolute.")
    resolved = path.resolve()
    if root is not None:
        resolved_root = root.resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise HandoffGateError(f"{label} binding escapes its authority root.")
    if not path.is_file() or path.is_symlink():
        raise HandoffGateError(f"{label} binding target is absent or unsafe.")
    if (
        path.stat().st_size != int(binding.get("size_bytes", -1))
        or _sha256_file(path) != binding.get("file_sha256")
    ):
        raise HandoffGateError(f"{label} byte binding drifted.")
    canonical: dict[str, Any] | None = None
    if "canonical_sha256" in binding:
        canonical = _load_digested(path, label=label)
        if canonical.get("sha256") != binding.get("canonical_sha256"):
            raise HandoffGateError(f"{label} canonical binding drifted.")
    return path, canonical


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
            "schema": "paper_i_matched_singleton12_live_runtime_fingerprint_v1",
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
            "schema": "paper_i_matched_singleton12_handoff_capacity_v1",
            "observed_at_utc": _utc_now(),
            "probe_path": probe.resolve().as_posix(),
            "available_memory_bytes": int(memory.available),
            "free_disk_bytes": int(disk.free),
        }
    )


def _load_source_runner() -> Any:
    if (
        not SOURCE_RUNNER.is_file()
        or SOURCE_RUNNER.is_symlink()
        or _sha256_file(SOURCE_RUNNER) != SOURCE_RUNNER_SHA256
    ):
        raise HandoffGateError("Pinned five-cell source runner is absent or drifted.")
    name = "paper_i_strong5_source_runner_for_matched12_handoff"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise HandoffGateError("Pinned five-cell source runner cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _try_existing_lock(path: Path) -> Any | None:
    if not path.is_file() or path.is_symlink():
        raise HandoffGateError(f"Required campaign lock is absent or unsafe: {path}")
    # BSD flock does not require a writable descriptor.  Opening read-only
    # keeps source inspection inert and lets preflight work in read-only
    # repository contexts.
    stream = path.open("r", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def _expected_source_status(
    source: Any,
    *,
    runtime: Mapping[str, Any],
    observed: Mapping[str, Any],
    status: str,
    terminal_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    expected = source._status_payload(
        runtime=runtime,
        status=status,
        completed=list(SOURCE_EXECUTION_IDS),
        current_execution_id=None,
        terminal_receipt_sha256=terminal_receipt_sha256,
    )
    return _digested(
        {
            **{key: value for key, value in expected.items() if key != "sha256"},
            "updated_at_utc": observed.get("updated_at_utc"),
        }
    )


def _validate_source_terminal(
    source: Any,
    *,
    source_lock: Any | None = None,
    repair_authorized: bool = False,
) -> dict[str, Any]:
    worker = source._load_worker()
    manifest, _rows = source._closed_inputs(worker)
    activation, plan, authorization, _remote_hold = source._validate_activation(
        SOURCE_ACTIVATION_DIR,
        manifest=manifest,
        require_fresh_hold=False,
    )
    runtime = source._ensure_runtime(SOURCE_RUNTIME_DIR, activation=activation)
    terminal = source._validate_terminal_receipt(
        activation_dir=SOURCE_ACTIVATION_DIR,
        runtime_dir=SOURCE_RUNTIME_DIR,
        activation=activation,
        plan=plan,
        authorization=authorization,
        runtime=runtime,
    )
    status_path = SOURCE_RUNTIME_DIR / "status/campaign.json"
    status_before = source._load_digested(
        status_path, label="source campaign status before closure"
    )
    before_binding = _binding(status_path, canonical=True)
    if (
        activation.get("sha256") != SOURCE_ACTIVATION_SHA256
        or runtime.get("sha256") != SOURCE_RUNTIME_SHA256
        or tuple(source.TARGET_EXECUTION_IDS) != SOURCE_EXECUTION_IDS
        or terminal.get("schema") != SOURCE_TERMINAL_SCHEMA
        or terminal.get("status") != SOURCE_TERMINAL_STATUS
        or terminal.get("execution_ids") != list(SOURCE_EXECUTION_IDS)
        or terminal.get("completed_execution_ids") != list(SOURCE_EXECUTION_IDS)
        or before_binding.get("canonical_sha256") != status_before.get("sha256")
    ):
        raise HandoffGateError("Five-cell immutable terminal closure drifted.")
    expected_final = _expected_source_status(
        source,
        runtime=runtime,
        observed=status_before,
        status=SOURCE_FINAL_STATUS,
        terminal_receipt_sha256=str(terminal["sha256"]),
    )
    expected_penultimate = _expected_source_status(
        source,
        runtime=runtime,
        observed=status_before,
        status="cell_passed_pending_remaining",
    )
    repair_performed = False
    final_status_pending = False
    status = status_before
    if status_before == expected_penultimate:
        if not repair_authorized:
            final_status_pending = True
        else:
            lock_path = SOURCE_RUNTIME_DIR / "campaign.lock"
            try:
                descriptor_stat = os.fstat(source_lock.fileno())
                lock_stat = lock_path.stat(follow_symlinks=False)
            except (AttributeError, OSError) as exc:
                raise HandoffGateError(
                    "Source final-status repair requires the acquired campaign lock."
                ) from exc
            if (
                source_lock is None
                or lock_path.is_symlink()
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (lock_stat.st_dev, lock_stat.st_ino)
            ):
                raise HandoffGateError(
                    "Source final-status repair requires the acquired campaign lock."
                )
            final = source._status_payload(
                runtime=runtime,
                status=SOURCE_FINAL_STATUS,
                completed=list(SOURCE_EXECUTION_IDS),
                current_execution_id=None,
                terminal_receipt_sha256=str(terminal["sha256"]),
            )
            source._write_json_atomic(status_path, final)
            status = source._load_digested(
                status_path, label="repaired source final campaign status"
            )
            if status != final:
                raise HandoffGateError(
                    "Repaired five-cell final campaign status drifted."
                )
            repair_performed = True
    elif status_before != expected_final:
        raise HandoffGateError("Five-cell immutable terminal closure drifted.")
    after_binding = _binding(status_path, canonical=True)
    if after_binding.get("canonical_sha256") != status.get("sha256"):
        raise HandoffGateError("Five-cell final status binding drifted.")
    status_repair = _digested(
        {
            "schema": SOURCE_STATUS_REPAIR_SCHEMA,
            "repair_performed": repair_performed,
            "before_status_binding": before_binding,
            "after_status_binding": after_binding,
        }
    )
    return {
        "terminal": terminal,
        "status": status,
        "terminal_binding": _binding(
            SOURCE_RUNTIME_DIR / "terminal_receipt.json", canonical=True
        ),
        "status_binding": _binding(status_path, canonical=True),
        "status_repair": status_repair,
        "final_status_pending": final_status_pending,
    }


def _probe_source(*, repair_authorized: bool = False) -> dict[str, Any]:
    if (
        not SOURCE_RUNNER.is_file()
        or SOURCE_RUNNER.is_symlink()
        or _sha256_file(SOURCE_RUNNER) != SOURCE_RUNNER_SHA256
    ):
        raise HandoffGateError("Pinned five-cell source runner drifted.")
    status_path = SOURCE_RUNTIME_DIR / "status/campaign.json"
    status = _load_digested(status_path, label="source campaign status")
    status_name = str(status.get("status"))
    if status_name in FAILURE_SOURCE_STATUSES:
        raise HandoffGateError(
            "Five-cell source campaign failed or was guard-stopped: "
            + json.dumps(status.get("failure"), sort_keys=True)
        )
    terminal_path = SOURCE_RUNTIME_DIR / "terminal_receipt.json"
    source_lock = _try_existing_lock(SOURCE_RUNTIME_DIR / "campaign.lock")
    if source_lock is None:
        if status_name not in ALLOWED_RUNNING_SOURCE_STATUSES and not terminal_path.exists():
            raise HandoffGateError(
                f"Unexpected active five-cell source status: {status_name}"
            )
        return {
            "state": "finalizing" if terminal_path.exists() else "running",
            "status": status,
        }
    try:
        if not terminal_path.is_file() or terminal_path.is_symlink():
            raise HandoffGateError(
                "Five-cell supervisor is inactive without an immutable terminal."
            )
        source = _load_source_runner()
        closure = _validate_source_terminal(
            source,
            source_lock=source_lock,
            repair_authorized=repair_authorized,
        )
        state = "finalizing" if closure["final_status_pending"] else "complete"
        return {"state": state, **closure}
    finally:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_UN)
        source_lock.close()


def _validate_source_prerequisite(value: Mapping[str, Any]) -> None:
    expected = {
        "runner_path": SOURCE_RUNNER.as_posix(),
        "runner_sha256": SOURCE_RUNNER_SHA256,
        "activation_dir": SOURCE_ACTIVATION_DIR.as_posix(),
        "activation_manifest_sha256": SOURCE_ACTIVATION_SHA256,
        "runtime_dir": SOURCE_RUNTIME_DIR.as_posix(),
        "runtime_manifest_sha256": SOURCE_RUNTIME_SHA256,
        "terminal_schema": SOURCE_TERMINAL_SCHEMA,
        "terminal_status": SOURCE_TERMINAL_STATUS,
        "final_status": SOURCE_FINAL_STATUS,
        "execution_ids": list(SOURCE_EXECUTION_IDS),
    }
    if dict(value) != expected:
        raise HandoffGateError("Target contract source prerequisite drifted.")


def _validate_cells(cells: Any) -> None:
    if not isinstance(cells, list) or len(cells) != 12:
        raise HandoffGateError("Target contract must contain exactly 12 cells.")
    identities: set[str] = set()
    observed: dict[tuple[str, int], set[str]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise HandoffGateError("Target cell entry is malformed.")
        execution_id = str(cell.get("execution_id", ""))
        method = str(cell.get("method", ""))
        regime = str(cell.get("regime", ""))
        try:
            n_ph = int(cell.get("n_ph"))
        except (TypeError, ValueError) as exc:
            raise HandoffGateError("Target cell n_ph is malformed.") from exc
        if not execution_id or execution_id in identities:
            raise HandoffGateError("Target cell execution IDs are absent or duplicated.")
        if method not in TARGET_METHODS or (regime, n_ph) not in TARGET_REGIMES:
            raise HandoffGateError("Target cell method/regime matrix drifted.")
        identities.add(execution_id)
        observed.setdefault((regime, n_ph), set()).add(method)
    if set(observed) != TARGET_REGIMES or any(
        methods != TARGET_METHODS for methods in observed.values()
    ):
        raise HandoffGateError("Target cells are not a matched six-regime 6+6 suite.")


def _validate_target_runtime_state(
    target: Mapping[str, Any], *, claim_exists: bool
) -> dict[str, Any]:
    runtime_dir = Path(str(target["runtime_dir"]))
    if runtime_dir.exists() or runtime_dir.is_symlink():
        if not runtime_dir.is_dir() or runtime_dir.is_symlink():
            raise HandoffGateError("Target runtime path is unsafe.")
        for name in ("in_progress", "quarantine"):
            path = runtime_dir / name
            if path.is_symlink() or (path.is_dir() and any(path.iterdir())):
                raise HandoffGateError(
                    f"Target {name} state requires manual inspection."
                )
        status_path = runtime_dir / "status/campaign.json"
        if status_path.exists() or status_path.is_symlink():
            status = _load_digested(status_path, label="target campaign status")
            if status.get("status") in {
                "failed",
                "failed_or_guard_stopped",
                "blocked",
            }:
                raise HandoffGateError("Target runtime records a failed state.")
        terminal_path = Path(str(target["expected_terminal"]["path"]))
        if terminal_path.exists() or terminal_path.is_symlink():
            if not claim_exists:
                raise HandoffGateError(
                    "Target terminal exists without an immutable handoff claim."
                )
            terminal = _load_digested(terminal_path, label="target terminal receipt")
            if (
                terminal.get("schema")
                != target["expected_terminal"]["schema"]
                or terminal.get("status")
                != target["expected_terminal"]["status"]
            ):
                raise HandoffGateError("Target terminal receipt drifted.")
            return {"state": "complete", "terminal": terminal}
        if not claim_exists:
            raise HandoffGateError(
                "Target runtime already exists before the handoff claim."
            )
        lock_path = runtime_dir / "campaign.lock"
        if lock_path.exists() or lock_path.is_symlink():
            lock = _try_existing_lock(lock_path)
            if lock is None:
                raise HandoffGateError("Target campaign is already active.")
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        return {"state": "replayable_after_claim"}
    return {"state": "absent"}


def _scientific_overlaps(markers: Sequence[Sequence[str]]) -> list[str]:
    matches: list[str] = []
    own_pid = os.getpid()
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = [str(item) for item in (process.info.get("cmdline") or [])]
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid == own_pid or not command:
            continue
        text = " ".join(command)
        if any(all(token in text for token in group) for group in markers):
            matches.append(f"{pid} {text}")
    return matches


def _load_target_contract(
    path: Path,
    *,
    claim_exists: bool,
    capacity_fn: Callable[[Path], dict[str, Any]] = _capacity,
    overlap_fn: Callable[[Sequence[Sequence[str]]], list[str]] = _scientific_overlaps,
) -> dict[str, Any]:
    contract = _load_digested(path, label="target handoff contract")
    if (
        contract.get("schema") != TARGET_CONTRACT_SCHEMA
        or contract.get("status") != TARGET_CONTRACT_STATUS
        or contract.get("execution_authorized") is not True
        or contract.get("submission_authorized") is not False
        or contract.get("paper_adoption_authorized") is not False
        or contract.get("paper_evidence_adoption_authorized") is not False
        or contract.get("gate_script_path") != GATE_PATH.as_posix()
        or contract.get("gate_script_sha256") != _sha256_file(GATE_PATH)
    ):
        raise HandoffGateError("Target handoff contract authority drifted.")
    source_prerequisite = contract.get("source_prerequisite")
    if not isinstance(source_prerequisite, Mapping):
        raise HandoffGateError("Target contract omits its source prerequisite.")
    _validate_source_prerequisite(source_prerequisite)
    target = contract.get("target")
    if not isinstance(target, Mapping):
        raise HandoffGateError("Target handoff contract omits target authority.")
    repo_root = Path(str(target.get("repo_root", "")))
    runner = Path(str(target.get("runner_path", "")))
    activation_dir = Path(str(target.get("activation_dir", "")))
    runtime_dir = Path(str(target.get("runtime_dir", "")))
    if (
        repo_root.resolve() != REPO_ROOT.resolve()
        or not runner.is_absolute()
        or not runner.is_file()
        or runner.is_symlink()
        or REPO_ROOT.resolve() not in runner.resolve().parents
        or _sha256_file(runner) != target.get("runner_sha256")
        or not activation_dir.is_absolute()
        or not activation_dir.is_dir()
        or activation_dir.is_symlink()
        or REPO_ROOT.resolve() not in activation_dir.resolve().parents
        or not runtime_dir.is_absolute()
        or REPO_ROOT.resolve() not in runtime_dir.resolve().parents
        or target.get("maximum_concurrency") != 1
    ):
        raise HandoffGateError("Target path, runner, or concurrency drifted.")
    expected_terminal = target.get("expected_terminal")
    if (
        not isinstance(expected_terminal, Mapping)
        or Path(str(expected_terminal.get("path", "")))
        != runtime_dir / "terminal_receipt.json"
        or not str(expected_terminal.get("schema", ""))
        or not str(expected_terminal.get("status", ""))
    ):
        raise HandoffGateError("Target terminal contract drifted.")
    command = target.get("command")
    expected_command = [
        PYTHON_EXECUTABLE.as_posix(),
        "-B",
        runner.as_posix(),
        "--run-campaign",
    ]
    if command != expected_command:
        raise HandoffGateError("Target command is not the fixed allowed command.")
    if target.get("environment") != REQUIRED_NUMERICAL_ENVIRONMENT:
        raise HandoffGateError("Target numerical environment drifted.")
    if (
        target.get("handoff_receipt_environment_variable")
        != HANDOFF_RECEIPT_ENV
        or target.get("handoff_token_environment_variable") != HANDOFF_TOKEN_ENV
        or target.get("handoff_lock_fd_environment_variable")
        != HANDOFF_LOCK_FD_ENV
    ):
        raise HandoffGateError("Target handoff environment seam drifted.")
    _validate_cells(target.get("cells"))
    authority_bindings = target.get("authority_bindings")
    required_bindings = {
        "planning_manifest",
        "execution_plan",
        "execution_authorization",
        "activation_manifest",
        "scientific_parity_canary",
        "runtime_fingerprint",
    }
    if not isinstance(authority_bindings, Mapping) or set(authority_bindings) != required_bindings:
        raise HandoffGateError("Target authority binding inventory drifted.")
    authority: dict[str, dict[str, Any]] = {}
    for name in sorted(required_bindings):
        _bound_path, payload = _validate_binding(
            authority_bindings[name], label=f"target {name}", root=activation_dir
        )
        if payload is None:
            raise HandoffGateError(f"Target {name} lacks a canonical binding.")
        authority[name] = payload
    authorization = authority["execution_authorization"]
    activation = authority["activation_manifest"]
    parity = authority["scientific_parity_canary"]
    if (
        authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("paper_adoption_authorized") is not False
        or activation.get("paper_evidence_adoption_authorized") is not False
        or parity.get("status") != target.get("expected_parity_status")
        or parity.get("paper_adoption_authorized") is not False
        or parity.get("paper_evidence_adoption_authorized") is not False
    ):
        raise HandoffGateError("Target activation or parity authority drifted.")
    live_runtime = _live_runtime_fingerprint()
    recorded_runtime = authority["runtime_fingerprint"]
    if recorded_runtime != live_runtime:
        raise HandoffGateError("Live Python/numerical runtime fingerprint drifted.")
    runtime_state = _validate_target_runtime_state(
        target, claim_exists=claim_exists
    )
    minimum_disk = int(target.get("minimum_free_disk_bytes", -1))
    minimum_memory = int(target.get("minimum_available_memory_bytes", -1))
    if (
        minimum_disk < HARD_MINIMUM_FREE_DISK_BYTES
        or minimum_memory < HARD_MINIMUM_AVAILABLE_MEMORY_BYTES
    ):
        raise HandoffGateError("Target capacity floors are below hard safety minima.")
    capacity = capacity_fn(runtime_dir)
    runtime_state_name = str(runtime_state.get("state"))
    if (
        runtime_state_name != "complete"
        and int(capacity.get("available_memory_bytes", -1)) < minimum_memory
    ):
        raise HandoffGateError(
            "Target launch memory capacity floor is not satisfied."
        )
    initial_claim = not claim_exists and runtime_state_name == "absent"
    if (
        initial_claim
        and int(capacity.get("free_disk_bytes", -1)) < minimum_disk
    ):
        raise HandoffGateError(
            "Target initial-claim disk capacity floor is not satisfied."
        )
    markers = target.get("scientific_overlap_markers")
    if (
        not isinstance(markers, list)
        or not markers
        or any(not isinstance(group, list) or not group for group in markers)
        or [runner.as_posix(), "--run-campaign"] not in markers
    ):
        raise HandoffGateError("Target scientific-overlap markers drifted.")
    combined_markers = [
        *[list(group) for group in BASE_SCIENTIFIC_OVERLAP_MARKERS],
        *markers,
    ]
    overlaps = (
        overlap_fn(combined_markers)
        if runtime_state_name != "complete"
        else []
    )
    if overlaps:
        raise HandoffGateError(
            "Another local scientific command overlaps the target: "
            + " | ".join(overlaps)
        )
    return {
        "contract": contract,
        "target": dict(target),
        "authority": authority,
        "live_runtime_fingerprint": live_runtime,
        "capacity": capacity,
        "runtime_state": runtime_state,
        "execution_authorized": True,
        "submission_authorized": False,
        "paper_adoption_authorized": False,
        "paper_evidence_adoption_authorized": False,
    }


def _deep_validate_completed_target_terminal(
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Import the exact contract-bound runner and invoke its read-only API."""

    runner = Path(str(target.get("runner_path", "")))
    repo_root = Path(str(target.get("repo_root", "")))
    expected_sha256 = str(target.get("runner_sha256", ""))
    try:
        before = runner.stat()
    except OSError as exc:
        raise HandoffGateError(
            "Completed-target runner is absent or unreadable."
        ) from exc
    if (
        not runner.is_absolute()
        or runner.is_symlink()
        or not runner.is_file()
        or repo_root.resolve() != REPO_ROOT.resolve()
        or REPO_ROOT.resolve() not in runner.resolve().parents
        or _sha256_file(runner) != expected_sha256
    ):
        raise HandoffGateError(
            "Completed-target runner path or byte binding drifted."
        )

    module_identity = hashlib.sha256(
        f"{runner.resolve().as_posix()}:{expected_sha256}".encode("utf-8")
    ).hexdigest()[:24]
    module_name = f"paper_i_matched12_completed_validator_{module_identity}"
    spec = importlib.util.spec_from_file_location(module_name, runner)
    if spec is None or spec.loader is None:
        raise HandoffGateError(
            "Completed-target runner cannot be imported."
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(str(module.__file__)).resolve() != runner.resolve():
            raise HandoffGateError(
                "Completed-target validator imported from the wrong path."
            )
        validator = getattr(
            module, "validate_completed_terminal_read_only", None
        )
        if not callable(validator):
            raise HandoffGateError(
                "Completed-target runner lacks its read-only terminal validator."
            )
        try:
            validated = validator()
        except Exception as exc:
            raise HandoffGateError(
                "Target runner deep terminal validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous

    try:
        after = runner.stat()
    except OSError as exc:
        raise HandoffGateError(
            "Completed-target runner disappeared during validation."
        ) from exc
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or _sha256_file(runner) != expected_sha256
    ):
        raise HandoffGateError(
            "Completed-target runner changed during deep validation."
        )
    if not isinstance(validated, Mapping):
        raise HandoffGateError(
            "Target runner deep terminal validator returned a non-mapping."
        )
    return dict(validated)


def _validate_completed_target_terminal(
    *, target_context: Mapping[str, Any], claim: Mapping[str, Any]
) -> dict[str, Any]:
    terminal = target_context["runtime_state"].get("terminal")
    target = target_context["target"]
    cells = target["cells"]
    execution_ids = [str(cell["execution_id"]) for cell in cells]
    authority = target_context["authority"]
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("execution_ids") != execution_ids
        or terminal.get("completed_execution_ids") != execution_ids
        or terminal.get("activation_manifest_sha256")
        != authority["activation_manifest"]["sha256"]
        or terminal.get("handoff_receipt_sha256") != claim.get("sha256")
        or terminal.get("execution_authorized") is not True
        or terminal.get("submission_authorized") is not False
        or terminal.get("paper_adoption_authorized") is not False
        or terminal.get("paper_evidence_adoption_authorized") is not False
        or claim.get("paper_adoption_authorized") is not False
        or claim.get("paper_evidence_adoption_authorized") is not False
    ):
        raise HandoffGateError(
            "Completed target terminal does not bind the authorized handoff."
        )
    terminal_path = Path(str(target["expected_terminal"]["path"]))
    terminal_binding = _binding(terminal_path, canonical=True)
    if terminal_binding.get("canonical_sha256") != terminal.get("sha256"):
        raise HandoffGateError(
            "Completed target terminal file binding drifted."
        )
    deeply_validated = _deep_validate_completed_target_terminal(target)
    if deeply_validated != dict(terminal):
        raise HandoffGateError(
            "Target runner deep terminal result does not exactly match the "
            "gate-loaded terminal."
        )
    return terminal_binding


def _status_payload(status: str, **fields: Any) -> dict[str, Any]:
    return _digested(
        {
            "schema": GATE_STATUS_SCHEMA,
            "status": status,
            "updated_at_utc": _utc_now(),
            **fields,
        }
    )


def _validated_source_status_repair(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    repair = source.get("status_repair")
    status_binding = source.get("status_binding")
    if not isinstance(repair, Mapping) or not isinstance(status_binding, Mapping):
        raise HandoffGateError("Source final-status repair evidence is absent.")
    before = repair.get("before_status_binding")
    after = repair.get("after_status_binding")
    performed = repair.get("repair_performed")
    if (
        repair.get("schema") != SOURCE_STATUS_REPAIR_SCHEMA
        or repair.get("sha256")
        != _canonical_sha256(
            {key: value for key, value in repair.items() if key != "sha256"}
        )
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or not isinstance(performed, bool)
        or dict(after) != dict(status_binding)
        or before.get("path") != after.get("path")
        or (performed is False and dict(before) != dict(after))
        or (performed is True and dict(before) == dict(after))
    ):
        raise HandoffGateError("Source final-status repair evidence drifted.")
    return dict(repair)


def _claim_payload(
    *,
    contract_path: Path,
    target_context: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    contract = target_context["contract"]
    target = target_context["target"]
    if (
        target_context.get("execution_authorized") is not True
        or target_context.get("submission_authorized") is not False
        or target_context.get("paper_adoption_authorized") is not False
        or target_context.get("paper_evidence_adoption_authorized") is not False
    ):
        raise HandoffGateError("Target context authority drifted.")
    source_status_repair = _validated_source_status_repair(source)
    command = list(target["command"])
    environment = dict(target["environment"])
    return _digested(
        {
            "schema": HANDOFF_RECEIPT_SCHEMA,
            "status": HANDOFF_RECEIPT_STATUS,
            "created_at_utc": _utc_now(),
            "gate_script": _binding(GATE_PATH),
            "target_contract": _binding(contract_path, canonical=True),
            "target_contract_sha256": contract["sha256"],
            "source_terminal": source["terminal_binding"],
            "source_terminal_sha256": source["terminal"]["sha256"],
            "source_final_status": source["status_binding"],
            "source_final_status_sha256": source["status"]["sha256"],
            "source_final_status_repair": source_status_repair,
            "target_authority_bindings": target["authority_bindings"],
            "target_activation_manifest_sha256": target_context["authority"][
                "activation_manifest"
            ]["sha256"],
            "target_scientific_parity_canary_sha256": target_context[
                "authority"
            ]["scientific_parity_canary"]["sha256"],
            "live_runtime_fingerprint_sha256": target_context[
                "live_runtime_fingerprint"
            ]["sha256"],
            "capacity": target_context["capacity"],
            "target_runtime_state": target_context["runtime_state"],
            "target_command": command,
            "target_command_sha256": hashlib.sha256(
                _canonical_json_bytes(command)
            ).hexdigest(),
            "target_environment": environment,
            "target_environment_sha256": hashlib.sha256(
                _canonical_json_bytes(environment)
            ).hexdigest(),
            "remote_runner_job_id": os.environ.get("REMOTE_JOB_ID"),
            "remote_runner_run_id": os.environ.get("REMOTE_RUN_ID"),
            "execution_authorized": True,
            "scientific_execution_performed": False,
            "submission_authorized": False,
            "submission_performed": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


@contextmanager
def _gate_lock(state_dir: Path) -> Iterator[Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "handoff.lock"
    if path.is_symlink():
        raise HandoffGateError("Handoff lock path is unsafe.")
    stream = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HandoffGateError("Another handoff gate owns the lock.") from exc
        yield stream
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def preflight(
    *,
    contract_path: Path = DEFAULT_TARGET_CONTRACT,
    source_probe: Callable[[], dict[str, Any]] = _probe_source,
) -> dict[str, Any]:
    contract_state = "absent"
    target_ready = False
    target_error: str | None = None
    if contract_path.exists() or contract_path.is_symlink():
        try:
            _load_target_contract(contract_path, claim_exists=False)
            contract_state = "validated_preauthorized"
            target_ready = True
        except (OSError, ValueError, KeyError, json.JSONDecodeError, HandoffGateError) as exc:
            contract_state = "blocked"
            target_error = str(exc)
    try:
        source = source_probe()
        source_state = str(source["state"])
        source_error = None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, HandoffGateError) as exc:
        source_state = "blocked"
        source_error = str(exc)
    return _digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed_inert_preflight",
            "observed_at_utc": _utc_now(),
            "source_state": source_state,
            "source_error": source_error,
            "target_contract_path": contract_path.as_posix(),
            "target_contract_state": contract_state,
            "target_error": target_error,
            "target_ready": target_ready,
            "run_ready": source_state == "complete" and target_ready,
            "live_runtime_fingerprint": _live_runtime_fingerprint(),
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def run_gate(
    *,
    contract_path: Path = DEFAULT_TARGET_CONTRACT,
    state_dir: Path = DEFAULT_STATE_DIR,
    source_probe: Callable[[], dict[str, Any]] | None = None,
    contract_loader: Callable[..., dict[str, Any]] = _load_target_contract,
    exec_fn: Callable[[str, Sequence[str], Mapping[str, str]], Any] = os.execve,
    sleep_fn: Callable[[float], Any] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    poll_seconds: float = POLL_SECONDS,
    wait_timeout_seconds: float = WAIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if poll_seconds <= 0 or wait_timeout_seconds <= 0:
        raise HandoffGateError("Gate polling and timeout must be positive.")
    status_path = state_dir / "status.json"
    claim_path = state_dir / "handoff_receipt.json"
    with _gate_lock(state_dir) as lock_stream:
        deadline = monotonic_fn() + wait_timeout_seconds
        while True:
            if monotonic_fn() > deadline:
                failed = _status_payload("blocked_timeout_waiting_for_handoff")
                _write_json_atomic(status_path, failed)
                raise HandoffGateError("Timed out waiting for the receipt-gated handoff.")
            try:
                source = (
                    _probe_source(repair_authorized=True)
                    if source_probe is None
                    else source_probe()
                )
                source_status_repair = (
                    _validated_source_status_repair(source)
                    if source.get("state") == "complete"
                    else None
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError, HandoffGateError) as exc:
                failed = _status_payload(
                    "blocked_source_failure", error_type=type(exc).__name__, error_message=str(exc)
                )
                _write_json_atomic(status_path, failed)
                raise
            if source.get("state") != "complete":
                _write_json_atomic(
                    status_path,
                    _status_payload(
                        "waiting_for_five_cell_immutable_terminal",
                        source_state=source.get("state"),
                    ),
                )
                sleep_fn(poll_seconds)
                continue
            if not contract_path.exists() and not contract_path.is_symlink():
                _write_json_atomic(
                    status_path,
                    _status_payload(
                        "waiting_for_preauthorized_target_contract",
                        source_terminal_sha256=source["terminal"]["sha256"],
                        source_final_status_repair=source_status_repair,
                        target_contract_path=contract_path.as_posix(),
                    ),
                )
                sleep_fn(poll_seconds)
                continue
            claim_exists = claim_path.exists() or claim_path.is_symlink()
            try:
                target_context = contract_loader(
                    contract_path, claim_exists=claim_exists
                )
                if claim_exists:
                    claim = _load_digested(claim_path, label="handoff receipt")
                    claimed_status_repair = _validated_source_status_repair(
                        {
                            "status_repair": claim.get(
                                "source_final_status_repair"
                            ),
                            "status_binding": source["status_binding"],
                        }
                    )
                    expected = _claim_payload(
                        contract_path=contract_path,
                        target_context=target_context,
                        source=source,
                    )
                    # Creation time, capacity observation, and remote-runner IDs
                    # are immutable observations from the first claim. Rebuild
                    # the payload with those observations before comparison.
                    expected = _digested(
                        {
                            **{k: v for k, v in expected.items() if k != "sha256"},
                            "created_at_utc": claim.get("created_at_utc"),
                            "capacity": claim.get("capacity"),
                            "target_runtime_state": claim.get(
                                "target_runtime_state"
                            ),
                            "source_final_status_repair": (
                                claimed_status_repair
                            ),
                            "remote_runner_job_id": claim.get("remote_runner_job_id"),
                            "remote_runner_run_id": claim.get("remote_runner_run_id"),
                        }
                    )
                    if claim != expected:
                        raise HandoffGateError("Existing immutable handoff receipt drifted.")
                    source_status_repair = claimed_status_repair
                else:
                    claim = _claim_payload(
                        contract_path=contract_path,
                        target_context=target_context,
                        source=source,
                    )
                    _write_json_atomic_noreplace(claim_path, claim)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, HandoffGateError) as exc:
                failed = _status_payload(
                    "blocked_target_validation",
                    source_terminal_sha256=source["terminal"]["sha256"],
                    source_final_status_repair=source_status_repair,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                _write_json_atomic(status_path, failed)
                raise
            if target_context["runtime_state"].get("state") == "complete":
                terminal_binding = _validate_completed_target_terminal(
                    target_context=target_context, claim=claim
                )
                completed = _status_payload(
                    "target_already_complete",
                    handoff_receipt_sha256=claim["sha256"],
                    target_terminal_sha256=target_context["runtime_state"][
                        "terminal"
                    ]["sha256"],
                    target_terminal_binding=terminal_binding,
                    source_final_status_repair=source_status_repair,
                )
                _write_json_atomic(status_path, completed)
                return completed
            _write_json_atomic(
                status_path,
                _status_payload(
                    "handoff_claimed_exec_pending",
                    source_terminal_sha256=source["terminal"]["sha256"],
                    handoff_receipt_sha256=claim["sha256"],
                    source_final_status_repair=source_status_repair,
                ),
            )
            token = hashlib.sha256(
                f"{claim['sha256']}:matched-singleton12-target-launch-v1".encode(
                    "utf-8"
                )
            ).hexdigest()
            os.set_inheritable(lock_stream.fileno(), True)
            environment = dict(os.environ)
            environment.update(target_context["target"]["environment"])
            environment.update(
                {
                    HANDOFF_RECEIPT_ENV: claim_path.as_posix(),
                    HANDOFF_TOKEN_ENV: token,
                    HANDOFF_LOCK_FD_ENV: str(lock_stream.fileno()),
                }
            )
            command = [str(item) for item in target_context["target"]["command"]]
            original_cwd = Path.cwd()
            os.chdir(target_context["target"]["repo_root"])
            try:
                exec_fn(command[0], command, environment)
            finally:
                # A successful exec never reaches this branch.  Restoring the
                # cwd makes an injected or unexpectedly returning exec seam
                # safe for focused tests and failure reporting.
                os.chdir(original_cwd)
            failed = _status_payload(
                "blocked_exec_returned_unexpectedly",
                handoff_receipt_sha256=claim["sha256"],
                source_final_status_repair=source_status_repair,
            )
            _write_json_atomic(status_path, failed)
            raise HandoffGateError("Target exec returned unexpectedly.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Receipt-gated Paper-I strong5 to matched-singleton12 handoff"
    )
    parser.add_argument(
        "--contract", type=Path, default=DEFAULT_TARGET_CONTRACT
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            payload = preflight(contract_path=args.contract.resolve())
        else:
            payload = run_gate(contract_path=args.contract.resolve())
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        psutil.Error,
        HandoffGateError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(_canonical_json_bytes(payload).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
