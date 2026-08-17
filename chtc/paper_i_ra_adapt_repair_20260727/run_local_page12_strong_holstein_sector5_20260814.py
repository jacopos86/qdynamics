#!/usr/bin/env python3
"""Run the five unfinished Page-12 strong-Holstein comparators locally.

The sealed twelve-cell package remains immutable.  This adapter binds exactly
the five held CHTC rows to a new local-only activation, overlays only the
tested checkpoint/resume memory repairs into an extracted source tree, and
runs one cell at a time behind RSS, available-memory, and disk guards.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import psutil


ADAPTER_PATH = Path(__file__).resolve()
REPAIR_ROOT = ADAPTER_PATH.parent
REPO_ROOT = ADAPTER_PATH.parents[2]
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
DEFAULT_PLANNING_DIR = REPAIR_ROOT / (
    "paper_i_page12_strong_holstein_sector5_local_repair_"
    "20260814_v1_planning"
)
DEFAULT_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_page12_strong_holstein_sector5_local_repair_"
    "20260814_v1_activation"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page12_strong_holstein_sector5_local_repair_20260814_v1"
)
DEFAULT_REMOTE_HOLD_RECEIPT = REPAIR_ROOT / (
    "paper_i_page12_cluster9647385_strong5_remote_hold_receipt_20260814.json"
)

PACKAGE_MANIFEST_FILE_SHA256 = (
    "4cf3df426f6b3545e51b90c0ffaf1f0755b989ae3644a067ca8cbcf98b5026bd"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "efce225efdc04653e8fca7e34eb3f467d4a6ec594e2130cde4bbea45e3d040e9"
)
SOURCE_ARCHIVE_SHA256 = (
    "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
)
QUEUE_FILE_SHA256 = (
    "406610e80d3f73521225da7852f1ee57414ee2ee1cfd0d4c73979d4b9f47527c"
)
TARGET_HORIZON = 50
MAXIMUM_CONCURRENCY = 1
CHECKPOINT_KEEP_HISTORY_TAIL = 1
RSS_LIMIT_BYTES = 12 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
MIN_LAUNCH_FREE_DISK_BYTES = 6 * 1024**3
RUNTIME_FREE_DISK_FLOOR_BYTES = 2 * 1024**3
GUARD_POLL_SECONDS = 1.0
STATUS_WRITE_SECONDS = 15.0
CHECKPOINT_TAIL_SCAN_BYTES = 16 * 1024**2
HOLD_RECEIPT_MAX_AGE_SECONDS = 30 * 60
LOCAL_CHILD_TOKEN_ENV = "PAPER_I_PAGE12_STRONG5_LOCAL_CHILD_TOKEN"
PARITY_CHILD_TOKEN_ENV = "PAPER_I_PAGE12_STRONG5_PARITY_CHILD_TOKEN"
LOCAL_EXECUTION_TARGET = "local_mac_guarded_serial_strong_holstein_sector5_v1"
PARITY_VARIANTS = ("sealed_baseline", "operational_candidate")
PARITY_MAXIMUM_ROUNDS = 1
PARITY_TIMEOUT_SECONDS = 15 * 60
SEALED_CURRENT_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)
CANDIDATE_CURRENT_CHECKPOINT_SHA256 = (
    "b6a0913ae2ee5f3dfd51ab99577980888a77a0cc01fd76bf5fe8437eab801535"
)
PARITY_JOB_SPEC_SHA256 = (
    "8b04e7f842e95efebeaaaacc80bce806e776b3fdc646d9b41721c2df0f034e1e"
)
PARITY_PROTOCOL_SHA256 = (
    "389d37fa55e1556faca9ef4c70d702b4029f4c4b75ad1c0a1621caa4661b9a79"
)
PARITY_ROUTE_SHA256 = (
    "24d5aed82ee202293187deb5e9745875a5779f8d6bca806536e4a323c7a307a6"
)
PARITY_BASELINE_ENERGY = 0.8768943743823379
LOCAL_NUMERICAL_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONHASHSEED": "0",
}

PLAN_SCHEMA = "paper_i_page12_strong_sector5_local_execution_plan_v1"
PLANNING_SCHEMA = "paper_i_page12_strong_sector5_local_planning_manifest_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_page12_strong_sector5_local_execution_authorization_v1"
)
ACTIVATION_SCHEMA = "paper_i_page12_strong_sector5_local_activation_v1"
PREFLIGHT_SCHEMA = "paper_i_page12_strong_sector5_local_preflight_v1"
RUNTIME_SCHEMA = "paper_i_page12_strong_sector5_local_runtime_v1"
STATUS_SCHEMA = "paper_i_page12_strong_sector5_local_status_v1"
GUARD_SCHEMA = "paper_i_page12_strong_sector5_local_guard_receipt_v1"
TERMINAL_SCHEMA = "paper_i_page12_strong_sector5_local_terminal_receipt_v1"
SOURCE_OVERLAY_SCHEMA = (
    "paper_i_page12_strong_sector5_local_source_overlay_receipt_v1"
)
REMOTE_HOLD_SCHEMA = "paper_i_page12_strong_sector5_remote_hold_receipt_v1"
PARITY_SPEC_SCHEMA = "paper_i_page12_strong_sector5_parity_spec_v1"
PARITY_BRANCH_SCHEMA = "paper_i_page12_strong_sector5_parity_branch_v1"
PARITY_SCHEMA = "paper_i_page12_strong_sector5_scientific_parity_v1"

OVERLAY_RELATIVE_PATHS = (
    Path("pipelines/static_adapt/current_checkpoint.py"),
)

HOLD_REASON = (
    "authorized local replacement guard for five unfinished strong-Holstein "
    "Paper-I cells (by user jsstrobel)"
)


class LocalStrongSectorError(RuntimeError):
    """Raised when a local strong-sector contract fails closed."""


@dataclass(frozen=True)
class TargetCell:
    proc: int
    regime_id: str
    policy: str
    hubbard_u: float
    exact_same_cutoff_energy: float

    @property
    def execution_id(self) -> str:
        return (
            "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
            f"{self.regime_id}__nph7__ra_global_singleton_gradient_phase0_"
            f"phase123_qiskit_phase23_{self.policy}"
        )


# Worst-case regimes run first so the memory/disk guards are exercised before
# the cheaper weak-strong cell.  Proc numbers are the original zero-based
# CHTC cluster 9647385 identities.
TARGET_CELLS = (
    TargetCell(
        5,
        "strong_strong_u8",
        "always_commutation_reduced",
        8.0,
        0.5205762765682245,
    ),
    TargetCell(
        11,
        "strong_strong_u8",
        "append_only",
        8.0,
        0.5205762765682245,
    ),
    TargetCell(
        4,
        "intermediate_strong",
        "always_commutation_reduced",
        1.25,
        -0.6239396137518985,
    ),
    TargetCell(
        10,
        "intermediate_strong",
        "append_only",
        1.25,
        -0.6239396137518985,
    ),
    TargetCell(
        3,
        "weak_strong",
        "always_commutation_reduced",
        0.25,
        -1.138720638075003,
    ),
)
TARGET_EXECUTION_IDS = tuple(cell.execution_id for cell in TARGET_CELLS)
TARGET_BY_ID = {cell.execution_id: cell for cell in TARGET_CELLS}
TARGET_PROCS = tuple(cell.proc for cell in TARGET_CELLS)

_WORKER: Any | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LocalStrongSectorError(f"{label} is not ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise LocalStrongSectorError(f"{label} must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


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
    if "sha256" in unsigned:
        raise LocalStrongSectorError("Cannot digest a payload twice.")
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise LocalStrongSectorError(f"Stale JSON temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic_noreplace(
    path: Path, payload: Mapping[str, Any]
) -> None:
    """Publish immutable JSON atomically without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_streaming(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical compact JSON without materializing the encoded bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for chunk in encoder.iterencode(payload):
            stream.write(chunk)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LocalStrongSectorError(f"{label} is absent or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LocalStrongSectorError(f"{label} must be a JSON object.")
    supplied = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise LocalStrongSectorError(f"{label} self-digest drifted.")
    return value


def _binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        result["canonical_sha256"] = _load_digested(
            path, label=path.name
        )["sha256"]
    return result


def _verify_binding(
    root: Path,
    raw: Any,
    *,
    expected_path: str,
    label: str,
    canonical: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or raw.get("path") != expected_path:
        raise LocalStrongSectorError(f"{label} binding path drifted.")
    path = root / expected_path
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or _sha256_file(path) != raw.get("sha256")
    ):
        raise LocalStrongSectorError(f"{label} byte binding drifted.")
    if canonical:
        value = _load_digested(path, label=label)
        if value["sha256"] != raw.get("canonical_sha256"):
            raise LocalStrongSectorError(f"{label} canonical binding drifted.")
        return value
    return None


def _load_worker() -> Any:
    global _WORKER
    if _WORKER is not None:
        return _WORKER
    package_text = PACKAGE_DIR.as_posix()
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    existing = sys.modules.get("package_contract")
    if existing is not None:
        existing_path = Path(str(getattr(existing, "__file__", "")))
        if existing_path.parent.resolve() != PACKAGE_DIR.resolve():
            del sys.modules["package_contract"]
    spec = importlib.util.spec_from_file_location(
        "paper_i_page12_strong5_sealed_worker",
        PACKAGE_DIR / "run_cell.py",
    )
    if spec is None or spec.loader is None:
        raise LocalStrongSectorError("Unable to load the sealed Page-12 worker.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _WORKER = module
    return module


def _queue_rows(worker: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue_path, _unused = worker._verify_binding(
        manifest.get("queue"), label="queue"
    )
    if worker.sha256_file(queue_path) != QUEUE_FILE_SHA256:
        raise LocalStrongSectorError("Sealed Page-12 queue bytes drifted.")
    rows: list[dict[str, Any]] = []
    for proc, line in enumerate(queue_path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 8:
            raise LocalStrongSectorError(f"Malformed queue row for proc {proc}.")
        (
            execution_id,
            job_path,
            protocol_path,
            job_file_sha256,
            request_cpus,
            request_memory_mb,
            request_disk_mb,
            max_runtime_seconds,
        ) = fields
        rows.append(
            {
                "proc": proc,
                "execution_id": execution_id,
                "job_path": job_path,
                "protocol_path": protocol_path,
                "job_file_sha256": job_file_sha256,
                "request_cpus": int(request_cpus),
                "request_memory_mb": int(request_memory_mb),
                "request_disk_mb": int(request_disk_mb),
                "max_runtime_seconds": int(max_runtime_seconds),
            }
        )
    if len(rows) != 12:
        raise LocalStrongSectorError("Sealed Page-12 queue row count drifted.")
    return rows


def _closed_inputs(
    worker: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    if worker.sha256_file(manifest_path) != PACKAGE_MANIFEST_FILE_SHA256:
        raise LocalStrongSectorError("Sealed package-manifest bytes drifted.")
    manifest = worker.load_json(manifest_path, label="package manifest")
    worker.verify_self_digest(manifest, label="package manifest")
    source = manifest.get("source_archive")
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("status") != "passed_inert_twelve_cells"
        or manifest.get("row_count") != 12
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise LocalStrongSectorError("Sealed Page-12 package identity drifted.")
    source_path, _unused = worker._verify_binding(
        source, label="source archive"
    )
    if worker.sha256_file(source_path) != SOURCE_ARCHIVE_SHA256:
        raise LocalStrongSectorError("Sealed source archive drifted.")
    rows = _queue_rows(worker, manifest)
    selected: list[dict[str, Any]] = []
    for cell in TARGET_CELLS:
        row = rows[cell.proc]
        if row["execution_id"] != cell.execution_id:
            raise LocalStrongSectorError(
                f"Proc-to-execution identity drifted at {cell.proc}."
            )
        job_path = PACKAGE_DIR / str(row["job_path"])
        if worker.sha256_file(job_path) != row["job_file_sha256"]:
            raise LocalStrongSectorError(f"Job file drifted: {cell.execution_id}")
        job, package, protocol, source_locks = worker._load_closed_job(job_path)
        if (
            package.get("sha256") != manifest.get("sha256")
            or job.get("execution_id") != cell.execution_id
            or job.get("regime_id") != cell.regime_id
            or int(job.get("nph", -1)) != 7
            or job.get("comparator_policy") != cell.policy
            or int(job.get("target_horizon", -1)) != TARGET_HORIZON
            or protocol.get("sha256") != job.get("protocol_sha256")
            or job.get("protocol_path") != row["protocol_path"]
            or protocol.get("request", {})
            .get("execution", {})
            .get("resume", {})
            .get("kind")
            != "fresh_start"
        ):
            raise LocalStrongSectorError(
                f"Sealed job/protocol drifted: {cell.execution_id}"
            )
        selected.append(
            {
                **row,
                "job": job,
                "protocol": protocol,
                "source_locks": source_locks,
            }
        )
    if tuple(row["execution_id"] for row in selected) != TARGET_EXECUTION_IDS:
        raise LocalStrongSectorError("Five-cell run order drifted.")
    return manifest, selected


def _sealed_worker_preflight(row: Mapping[str, Any]) -> dict[str, Any]:
    """Run one sealed preflight in a child so imports cannot escape."""

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            (PACKAGE_DIR / "run_cell.py").as_posix(),
            "--job",
            (PACKAGE_DIR / str(row["job_path"])).as_posix(),
            "--preflight",
        ],
        cwd=PACKAGE_DIR,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise LocalStrongSectorError(
            f"Sealed preflight failed for {row['execution_id']}: "
            f"{completed.stderr.strip()}"
        )
    try:
        receipt = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LocalStrongSectorError(
            f"Sealed preflight emitted invalid JSON: {row['execution_id']}"
        ) from exc
    if not isinstance(receipt, dict):
        raise LocalStrongSectorError("Sealed preflight receipt is not a mapping.")
    return receipt


def _overlay_bindings() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in OVERLAY_RELATIVE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise LocalStrongSectorError(f"Operational overlay is unsafe: {path}")
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def _validate_remote_hold_receipt(
    path: Path,
    *,
    require_fresh: bool,
) -> dict[str, Any]:
    receipt = _load_digested(path, label="authenticated remote hold receipt")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        raise LocalStrongSectorError("Remote hold row inventory is absent.")
    expected_by_proc = {
        cell.proc: cell.execution_id for cell in TARGET_CELLS
    }
    observed_by_proc: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise LocalStrongSectorError("Remote hold row is malformed.")
        try:
            proc = int(row.get("proc", -1))
        except (TypeError, ValueError) as exc:
            raise LocalStrongSectorError("Remote hold proc is malformed.") from exc
        execution_id = str(row.get("execution_id", ""))
        if (
            proc in observed_by_proc
            or expected_by_proc.get(proc) != execution_id
            or int(row.get("job_status", -1)) != 5
            or row.get("hold_reason") != HOLD_REASON
        ):
            raise LocalStrongSectorError("Remote hold row identity drifted.")
        observed_by_proc[proc] = execution_id
    observed_at = _utc_datetime(
        receipt.get("observed_at_utc"), label="remote hold observed_at_utc"
    )
    now = datetime.now(timezone.utc)
    age_seconds = (now - observed_at).total_seconds()
    expected_rows_digest = _canonical_sha256({"rows": rows})
    if (
        receipt.get("schema") != REMOTE_HOLD_SCHEMA
        or receipt.get("status") != "passed_authenticated_exact_remote_holds"
        or receipt.get("scheduler") != "chtc_condor"
        or receipt.get("cluster_id") != 9647385
        or receipt.get("authentication_kind")
        != "interactive_ssh_duo_condor_hold_query_v1"
        or receipt.get("authenticated_remote_query") is not True
        or observed_by_proc != expected_by_proc
        or receipt.get("held_procs") != sorted(TARGET_PROCS)
        or set(receipt.get("held_execution_ids", ()))
        != set(TARGET_EXECUTION_IDS)
        or receipt.get("remote_active_execution_ids") != []
        or receipt.get("late_materialization_factory_active") is not False
        or receipt.get("remote_rows_sha256") != expected_rows_digest
        or receipt.get("scientific_execution_performed") is not False
        or age_seconds < 0
        or (require_fresh and age_seconds > HOLD_RECEIPT_MAX_AGE_SECONDS)
    ):
        raise LocalStrongSectorError("Authenticated remote hold receipt drifted.")
    return receipt


def _copy_overlays(source_root: Path) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for binding in _overlay_bindings():
        relative = Path(str(binding["path"]))
        source = REPO_ROOT / relative
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if (
            destination.is_symlink()
            or _sha256_file(destination) != binding["sha256"]
            or destination.stat().st_size != binding["size_bytes"]
        ):
            raise LocalStrongSectorError(f"Source overlay copy drifted: {relative}")
        applied.append(dict(binding))
    return applied


def _overlay_protocol_preflight(execution_id: str) -> dict[str, Any]:
    """Materialize the repaired source/protocol without scientific execution."""

    if execution_id not in TARGET_BY_ID:
        raise LocalStrongSectorError("Overlay preflight is outside the five-cell scope.")
    worker = _load_worker()
    manifest, rows = _closed_inputs(worker)
    row = next(row for row in rows if row["execution_id"] == execution_id)
    temporary = tempfile.TemporaryDirectory(prefix="paper-i-strong5-overlay-preflight.")
    try:
        source_root = Path(temporary.name) / "source"
        worker._extract_source(
            manifest=manifest,
            source_locks=row["source_locks"],
            destination=source_root,
        )
        applied = _copy_overlays(source_root)
        original_cwd = Path.cwd()
        os.chdir(source_root)
        try:
            worker._activate_source_root(source_root)
            protocol, _problem = worker._load_protocol(
                job=row["job"],
                payload=row["protocol"],
                source_locks=row["source_locks"],
            )
        finally:
            os.chdir(original_cwd)
        if (
            protocol.sha256 != row["job"]["protocol_sha256"]
            or protocol.route_contract["sha256"]
            != row["job"]["route_contract_sha256"]
        ):
            raise LocalStrongSectorError("Overlay protocol preflight drifted.")
        return _digested(
            {
                "schema": "paper_i_page12_strong_sector5_overlay_preflight_v1",
                "status": "passed_inert_overlay_protocol_preflight",
                "execution_id": execution_id,
                "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "overlay_files": applied,
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "scientific_execution_performed": False,
            }
        )
    finally:
        temporary.cleanup()


def _isolated_overlay_preflight(execution_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            ADAPTER_PATH.as_posix(),
            "--overlay-preflight",
            execution_id,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise LocalStrongSectorError(
            "Operational-overlay preflight failed: " + completed.stderr.strip()
        )
    value = json.loads(completed.stdout)
    if (
        not isinstance(value, dict)
        or value.get("status") != "passed_inert_overlay_protocol_preflight"
        or value.get("execution_id") != execution_id
        or value.get("scientific_execution_performed") is not False
    ):
        raise LocalStrongSectorError("Operational-overlay preflight receipt drifted.")
    return value


def _capacity(runtime_dir: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(runtime_dir.parent)
    blockers: list[str] = []
    if int(memory.available) < AVAILABLE_MEMORY_FLOOR_BYTES:
        blockers.append("available_memory_below_launch_floor")
    if int(disk.free) < MIN_LAUNCH_FREE_DISK_BYTES:
        blockers.append("free_disk_below_launch_floor")
    return _digested(
        {
            "schema": "paper_i_page12_strong_sector5_local_capacity_v1",
            "status": "passed" if not blockers else "blocked",
            "physical_memory_bytes": int(memory.total),
            "available_memory_bytes": int(memory.available),
            "free_disk_bytes": int(disk.free),
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
            "minimum_launch_free_disk_bytes": MIN_LAUNCH_FREE_DISK_BYTES,
            "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "blockers": blockers,
            "scientific_execution_performed": False,
        }
    )


def _cell_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    job = row["job"]
    protocol = row["protocol"]
    cell = TARGET_BY_ID[str(row["execution_id"])]
    return {
        "cluster_id": 9647385,
        "proc": int(row["proc"]),
        "execution_id": row["execution_id"],
        "regime_id": cell.regime_id,
        "nph": 7,
        "hubbard_u": cell.hubbard_u,
        "electron_phonon_coupling": 0.790569415042,
        "exact_same_cutoff_energy": cell.exact_same_cutoff_energy,
        "comparator_policy": cell.policy,
        "job_path": row["job_path"],
        "job_file_sha256": row["job_file_sha256"],
        "job_spec_sha256": job["sha256"],
        "protocol_path": row["protocol_path"],
        "protocol_file_sha256": job["protocol_file_sha256"],
        "protocol_sha256": protocol["sha256"],
        "route_contract_sha256": job["route_contract_sha256"],
        "target_horizon": TARGET_HORIZON,
        "optimizer": "POWELL",
        "optimizer_maxiter": 200,
        "adapt_seed": 7,
        "transpiler_seed": 7,
        "fresh_start": True,
        "sealed_scheduler_resources": {
            "request_cpus": int(row["request_cpus"]),
            "request_memory_mb": int(row["request_memory_mb"]),
            "request_disk_mb": int(row["request_disk_mb"]),
            "max_runtime_seconds": int(row["max_runtime_seconds"]),
            "provenance_only_not_local_reservation": True,
        },
    }


def _local_numerical_runtime_policy() -> dict[str, Any]:
    return {
        "platform": "local_macos_arm64_host_python",
        "thread_environment": dict(LOCAL_NUMERICAL_THREAD_ENVIRONMENT),
        "same_runtime_required_across_all_five_cells": True,
        "same_local_runtime_required_for_parity_canary": True,
        "historical_chtc_generator_sequence_parity_required": False,
        "historical_symmetry_tie_branch_is_not_scientific_target": True,
        "future_append_ra_direct_comparison_requires_matched_runtime": True,
    }


def _parity_canary_spec() -> dict[str, Any]:
    if TARGET_CELLS[0].proc != 5:
        raise LocalStrongSectorError("Parity canary is not pinned to proc 5.")
    overlays = _overlay_bindings()
    if (
        len(overlays) != 1
        or overlays[0]["path"]
        != "pipelines/static_adapt/current_checkpoint.py"
        or overlays[0]["sha256"] != CANDIDATE_CURRENT_CHECKPOINT_SHA256
    ):
        raise LocalStrongSectorError("Parity candidate overlay identity drifted.")
    return _digested(
        {
            "schema": PARITY_SPEC_SCHEMA,
            "status": "required_before_activation",
            "execution_id": TARGET_EXECUTION_IDS[0],
            "cluster_id": 9647385,
            "proc": 5,
            "maximum_controller_rounds": PARITY_MAXIMUM_ROUNDS,
            "variants": list(PARITY_VARIANTS),
            "comparison": "exact_canonical_scientific_projection_v1",
            "excluded_operational_result_fields": [
                "run.observation",
                "scientific_receipts.controller_replay_evidence",
                "scientific_receipts.controller_replay_evidence_sha256",
            ],
            "normalized_operational_result_fields": [
                "run.paper_i_summary.append_matched.failure.message:"
                "authorized_parity_temporary_root_only"
            ],
            "job_spec_sha256": PARITY_JOB_SPEC_SHA256,
            "protocol_sha256": PARITY_PROTOCOL_SHA256,
            "route_contract_sha256": PARITY_ROUTE_SHA256,
            "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "sealed_current_checkpoint_sha256": (
                SEALED_CURRENT_CHECKPOINT_SHA256
            ),
            "candidate_current_checkpoint_sha256": (
                CANDIDATE_CURRENT_CHECKPOINT_SHA256
            ),
            "candidate_overlay_files": overlays,
            "diagnostic_only": True,
            "campaign_cell_progress_credited": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _scientific_parity_projection(
    result_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project every scientific result field, excluding checkpoint transport."""

    expected_top_level = {
        "schema",
        "protocol",
        "selector_identity",
        "parent_inventory",
        "executable_pool",
        "policy",
        "run",
        "numerical_physical_integrity",
        "scientific_receipts",
    }
    if set(result_payload) != expected_top_level:
        raise LocalStrongSectorError("Parity result top-level schema drifted.")
    projection = json.loads(
        _canonical_json_bytes(result_payload).decode("utf-8")
    )
    run = projection.get("run")
    receipts = projection.get("scientific_receipts")
    if not isinstance(run, dict) or not isinstance(receipts, dict):
        raise LocalStrongSectorError("Parity result projection is malformed.")
    observation = run.pop("observation", None)
    replay_evidence = receipts.pop("controller_replay_evidence", None)
    replay_sha = receipts.pop("controller_replay_evidence_sha256", None)
    if (
        not isinstance(observation, dict)
        or not isinstance(replay_evidence, dict)
        or not isinstance(replay_sha, str)
    ):
        raise LocalStrongSectorError(
            "Parity operational checkpoint evidence is absent."
        )
    summary = run.get("paper_i_summary")
    if isinstance(summary, dict):
        append_matched = summary.get("append_matched")
        if isinstance(append_matched, dict):
            failure = append_matched.get("failure")
            if isinstance(failure, dict) and isinstance(
                failure.get("message"), str
            ):
                failure["message"] = re.sub(
                    (
                        r"/[^']*/paper-i-strong5-parity-"
                        r"(?:sealed_baseline|operational_candidate)\.[^/']+/"
                        r"paper-i-strong5-"
                        r"(?:sealed_baseline|operational_candidate)\.[^/']+"
                    ),
                    "<authorized-parity-temporary-root>",
                    failure["message"],
                )
    return projection


def _first_canonical_difference(
    left: Any, right: Any, *, path: str = "$"
) -> str | None:
    if type(left) is not type(right):
        return f"{path}:type:{type(left).__name__}!={type(right).__name__}"
    if isinstance(left, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return (
                f"{path}:keys:"
                f"left_only={sorted(left_keys - right_keys)},"
                f"right_only={sorted(right_keys - left_keys)}"
            )
        for key in sorted(left_keys):
            difference = _first_canonical_difference(
                left[key], right[key], path=f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}:length:{len(left)}!={len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_canonical_difference(
                left_item, right_item, path=f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        left_text = repr(left)
        right_text = repr(right)
        return f"{path}:value:{left_text[:180]}!={right_text[:180]}"
    return None


def prepare_planning(
    *,
    planning_dir: Path,
    runtime_dir: Path,
    remote_hold_receipt: Path = DEFAULT_REMOTE_HOLD_RECEIPT,
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_inputs(worker)
    remote_hold = _validate_remote_hold_receipt(
        remote_hold_receipt, require_fresh=True
    )
    if planning_dir.exists() or planning_dir.is_symlink():
        raise FileExistsError(f"Planning destination exists: {planning_dir}")
    capacity = _capacity(runtime_dir)
    preflights = []
    for row in rows:
        receipt = _sealed_worker_preflight(row)
        if (
            receipt.get("status") != "passed"
            or receipt.get("execution_id") != row["execution_id"]
            or receipt.get("fresh_start") is not True
            or int(receipt.get("target_horizon", -1)) != TARGET_HORIZON
            or receipt.get("scientific_execution_performed") is not False
        ):
            raise LocalStrongSectorError(
                f"Sealed preflight drifted: {row['execution_id']}"
            )
        preflights.append(receipt)
    overlay_preflight = _isolated_overlay_preflight(TARGET_EXECUTION_IDS[0])
    planning_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{planning_dir.name}.build-",
            dir=planning_dir.parent,
        )
    )
    try:
        adapter_sha256 = _sha256_file(ADAPTER_PATH)
        copied_hold_path = temporary / "remote_hold_receipt.json"
        _write_json_exclusive(copied_hold_path, remote_hold)
        hold_binding = _binding(
            copied_hold_path, root=temporary, canonical=True
        )
        plan = _digested(
            {
                "schema": PLAN_SCHEMA,
                "status": "planned_immutable_not_authorized",
                "created_at_utc": _utc_now(),
                "run_class": "candidate_compatibility_comparator",
                "paper_scope": "Paper-I Page-12 insertion comparator",
                "scope": "five_unfinished_strong_holstein_sector_cells_only",
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "package_manifest_sha256": manifest["sha256"],
                "package_manifest_file_sha256": PACKAGE_MANIFEST_FILE_SHA256,
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": adapter_sha256,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "run_order": list(TARGET_EXECUTION_IDS),
                "cells": [_cell_plan_row(row) for row in rows],
                "scientific_settings": {
                    "sites": 2,
                    "qubits": 10,
                    "boundary": "open",
                    "phonon_encoding": "binary",
                    "phonon_layout": "blocked",
                    "hopping_t": 1.0,
                    "phonon_frequency": 1.0,
                    "nph": 7,
                    "target_horizon": TARGET_HORIZON,
                    "optimizer": "POWELL",
                    "optimizer_maxiter": 200,
                    "adapt_seed": 7,
                    "transpiler_seed": 7,
                },
                "numerical_runtime_policy": _local_numerical_runtime_policy(),
                "operational_repairs": {
                    "source_overlay_files": _overlay_bindings(),
                    "checkpoint_keep_history_tail": (
                        CHECKPOINT_KEEP_HISTORY_TAIL
                    ),
                    "stream_final_json_encoding": True,
                    "ai_log_suppressed": True,
                    "scientific_protocol_settings_changed": False,
                    "route_contracts_changed": False,
                    "observation_serialization_only": True,
                },
                "guards": {
                    "maximum_concurrency": MAXIMUM_CONCURRENCY,
                    "rss_limit_bytes": RSS_LIMIT_BYTES,
                    "available_memory_floor_bytes": (
                        AVAILABLE_MEMORY_FLOOR_BYTES
                    ),
                    "minimum_launch_free_disk_bytes": (
                        MIN_LAUNCH_FREE_DISK_BYTES
                    ),
                    "runtime_free_disk_floor_bytes": (
                        RUNTIME_FREE_DISK_FLOOR_BYTES
                    ),
                },
                "scientific_parity_canary_spec": _parity_canary_spec(),
                "remote_overlap_control": {
                    "remote_hold_receipt": hold_binding,
                    "remote_hold_receipt_sha256": remote_hold["sha256"],
                    "scheduler": remote_hold["scheduler"],
                    "cluster_id": remote_hold["cluster_id"],
                    "held_procs": remote_hold["held_procs"],
                    "observed_at_utc": remote_hold["observed_at_utc"],
                    "authenticated_remote_query": True,
                    "remote_release_authorized": False,
                },
                "source_authorizations_reused": False,
                "execution_authorized": False,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        plan_path = temporary / "execution_plan.json"
        _write_json_exclusive(plan_path, plan)
        plan_binding = _binding(plan_path, root=temporary, canonical=True)

        preflight = _digested(
            {
                "schema": PREFLIGHT_SCHEMA,
                "status": "passed_inert_activation_preflight",
                "created_at_utc": _utc_now(),
                "package_manifest_sha256": manifest["sha256"],
                "local_adapter_sha256": adapter_sha256,
                "sealed_worker_preflights": preflights,
                "operational_overlay_preflight": overlay_preflight,
                "remote_hold_receipt": hold_binding,
                "capacity_at_activation": capacity,
                "capacity_recheck_required_before_each_cell": True,
                "scientific_execution_performed": False,
            }
        )
        preflight_path = temporary / "host_preflight.json"
        _write_json_exclusive(preflight_path, preflight)

        planning = _digested(
            {
                "schema": PLANNING_SCHEMA,
                "status": "passed_local_plan_not_authorized",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": adapter_sha256,
                "execution_plan": plan_binding,
                "host_preflight": _binding(
                    preflight_path, root=temporary, canonical=True
                ),
                "remote_hold_receipt": hold_binding,
                "execution_ids": list(TARGET_EXECUTION_IDS),
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": False,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json_exclusive(
            temporary / "planning_manifest.json", planning
        )
        os.rename(temporary, planning_dir)
        return planning
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_planning(
    planning_dir: Path,
    *,
    manifest: Mapping[str, Any],
    require_fresh_hold: bool = False,
    reject_authority_files: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if not planning_dir.is_dir() or planning_dir.is_symlink():
        raise LocalStrongSectorError("Local planning directory is absent or unsafe.")
    if reject_authority_files and any(
        path.exists() or path.is_symlink()
        for path in (
            planning_dir / "execution_authorization.json",
            planning_dir / "activation_manifest.json",
        )
    ):
        raise LocalStrongSectorError(
            "Inert planning directory contains execution authority."
        )
    planning = _load_digested(
        planning_dir / "planning_manifest.json",
        label="local planning manifest",
    )
    plan = _verify_binding(
        planning_dir,
        planning.get("execution_plan"),
        expected_path="execution_plan.json",
        label="execution plan",
        canonical=True,
    )
    preflight = _verify_binding(
        planning_dir,
        planning.get("host_preflight"),
        expected_path="host_preflight.json",
        label="host preflight",
        canonical=True,
    )
    remote_hold = _verify_binding(
        planning_dir,
        planning.get("remote_hold_receipt"),
        expected_path="remote_hold_receipt.json",
        label="authenticated remote hold receipt",
        canonical=True,
    )
    assert (
        plan is not None
        and preflight is not None
        and remote_hold is not None
    )
    remote_hold = _validate_remote_hold_receipt(
        planning_dir / "remote_hold_receipt.json",
        require_fresh=require_fresh_hold,
    )
    expected_overlay = _overlay_bindings()
    if (
        planning.get("schema") != PLANNING_SCHEMA
        or planning.get("status") != "passed_local_plan_not_authorized"
        or planning.get("package_manifest_sha256") != manifest.get("sha256")
        or planning.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or planning.get("local_adapter_sha256") != _sha256_file(ADAPTER_PATH)
        or planning.get("execution_ids") != list(TARGET_EXECUTION_IDS)
        or planning.get("execution_target") != LOCAL_EXECUTION_TARGET
        or planning.get("execution_authorized") is not False
        or planning.get("submission_authorized") is not False
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "planned_immutable_not_authorized"
        or plan.get("execution_authorized") is not False
        or plan.get("run_order") != list(TARGET_EXECUTION_IDS)
        or plan.get("remote_overlap_control", {}).get(
            "remote_hold_receipt"
        )
        != planning.get("remote_hold_receipt")
        or plan.get("remote_overlap_control", {}).get(
            "remote_hold_receipt_sha256"
        )
        != remote_hold.get("sha256")
        or plan.get("operational_repairs", {}).get("source_overlay_files")
        != expected_overlay
        or plan.get("operational_repairs", {}).get(
            "checkpoint_keep_history_tail"
        )
        != CHECKPOINT_KEEP_HISTORY_TAIL
        or plan.get("scientific_parity_canary_spec")
        != _parity_canary_spec()
        or plan.get("numerical_runtime_policy")
        != _local_numerical_runtime_policy()
        or preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("remote_hold_receipt", {}).get("canonical_sha256")
        != remote_hold.get("sha256")
        or preflight.get("scientific_execution_performed") is not False
    ):
        raise LocalStrongSectorError("Local planning contract drifted.")
    return planning, plan, preflight, remote_hold


def _expected_parity_child_token(
    authorization: Mapping[str, Any], variant: str
) -> str:
    return hashlib.sha256(
        f"{authorization['sha256']}:{variant}:parity-child-v1".encode("utf-8")
    ).hexdigest()


def _validate_inline_digested(
    value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LocalStrongSectorError(f"{label} must be a JSON object.")
    supplied = value.get("sha256")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if supplied != _canonical_sha256(unsigned):
        raise LocalStrongSectorError(f"{label} self-digest drifted.")
    return value


def _run_parity_branch(
    *, variant: str, activation_dir: Path
) -> dict[str, Any]:
    if variant not in PARITY_VARIANTS:
        raise LocalStrongSectorError("Parity branch variant is outside scope.")
    authorization = _load_digested(
        activation_dir / "execution_authorization.json",
        label="parity execution authorization",
    )
    plan = _verify_binding(
        activation_dir,
        authorization.get("execution_plan"),
        expected_path="execution_plan.json",
        label="parity execution plan",
        canonical=True,
    )
    assert plan is not None
    spec = plan.get("scientific_parity_canary_spec")
    if (
        spec != _parity_canary_spec()
        or authorization.get("scientific_parity_canary_authorized") is not True
        or authorization.get("parity_canary_spec_sha256")
        != spec.get("sha256")
        or authorization.get("parity_canary_variants")
        != list(PARITY_VARIANTS)
        or authorization.get("parity_canary_maximum_rounds")
        != PARITY_MAXIMUM_ROUNDS
        or os.environ.get(PARITY_CHILD_TOKEN_ENV)
        != _expected_parity_child_token(authorization, variant)
    ):
        raise LocalStrongSectorError("Parity branch authority drifted.")

    worker = _load_worker()
    manifest, rows = _closed_inputs(worker)
    execution_id = TARGET_EXECUTION_IDS[0]
    row = next(item for item in rows if item["execution_id"] == execution_id)
    if (
        int(row["proc"]) != 5
        or row["job"]["sha256"] != PARITY_JOB_SPEC_SHA256
        or row["job"]["protocol_sha256"] != PARITY_PROTOCOL_SHA256
        or row["job"]["route_contract_sha256"] != PARITY_ROUTE_SHA256
    ):
        raise LocalStrongSectorError("Parity sealed cell identity drifted.")

    with tempfile.TemporaryDirectory(prefix=f"paper-i-strong5-{variant}.") as raw:
        temporary = Path(raw)
        source_root = temporary / "source"
        staging = temporary / "cell_output"
        worker._extract_source(
            manifest=manifest,
            source_locks=row["source_locks"],
            destination=source_root,
        )
        checkpoint_source = (
            source_root / "pipelines/static_adapt/current_checkpoint.py"
        )
        if _sha256_file(checkpoint_source) != SEALED_CURRENT_CHECKPOINT_SHA256:
            raise LocalStrongSectorError("Parity sealed checkpoint source drifted.")
        if variant == "operational_candidate":
            applied = _copy_overlays(source_root)
            if (
                len(applied) != 1
                or _sha256_file(checkpoint_source)
                != CANDIDATE_CURRENT_CHECKPOINT_SHA256
            ):
                raise LocalStrongSectorError("Parity candidate overlay drifted.")
        source_checkpoint_sha256 = _sha256_file(checkpoint_source)
        staging.mkdir()
        original_cwd = Path.cwd()
        os.chdir(source_root)
        try:
            worker._activate_source_root(source_root)
            protocol, problem = worker._load_protocol(
                job=row["job"],
                payload=row["protocol"],
                source_locks=row["source_locks"],
            )
            if (
                protocol.sha256 != PARITY_PROTOCOL_SHA256
                or protocol.route_contract["sha256"] != PARITY_ROUTE_SHA256
            ):
                raise LocalStrongSectorError("Parity protocol drifted.")
            from pipelines.static_adapt import adapt_pipeline
            import pipelines.static_adapt.sr_snake as sr_snake

            adapt_pipeline._ai_log = lambda *args, **kwargs: None
            original_checkpoint_observation = sr_snake.CheckpointObservation
            if variant == "operational_candidate":
                sr_snake.CheckpointObservation = (
                    _compact_checkpoint_observation_factory(
                        original_checkpoint_observation
                    )
                )
            try:
                result, rounds = worker._execute(
                    protocol=protocol,
                    problem=problem,
                    staging=staging,
                    maximum_rounds=PARITY_MAXIMUM_ROUNDS,
                )
            finally:
                sr_snake.CheckpointObservation = original_checkpoint_observation
        finally:
            os.chdir(original_cwd)
        if rounds != PARITY_MAXIMUM_ROUNDS:
            raise LocalStrongSectorError("Parity branch did not close one round.")
        result_payload = result.to_dict()
        if not isinstance(result_payload, Mapping):
            raise LocalStrongSectorError("Parity branch result is malformed.")
        projection = _scientific_parity_projection(result_payload)
        run_payload = result_payload.get("run")
        if not isinstance(run_payload, Mapping):
            raise LocalStrongSectorError("Parity run payload is absent.")
        final_state = run_payload.get("final_state")
        transitions = run_payload.get("accepted_transitions")
        observation = run_payload.get("observation")
        if (
            not isinstance(final_state, Mapping)
            or not isinstance(transitions, list)
            or len(transitions) != 1
            or not isinstance(transitions[0], Mapping)
            or not isinstance(observation, Mapping)
        ):
            raise LocalStrongSectorError("Parity one-round witness drifted.")
        ledger_path = staging / "result/estimator_ledger.json"
        checkpoint_path = staging / "checkpoints/current.json"
        if not ledger_path.is_file() or not checkpoint_path.is_file():
            raise LocalStrongSectorError("Parity branch artifacts are absent.")
        return _digested(
            {
                "schema": PARITY_BRANCH_SCHEMA,
                "status": "passed_authorized_diagnostic_branch",
                "variant": variant,
                "execution_id": execution_id,
                "controller_rounds_completed": rounds,
                "job_spec_sha256": row["job"]["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "current_checkpoint_source_sha256": (
                    source_checkpoint_sha256
                ),
                "execution_plan_sha256": plan["sha256"],
                "execution_authorization_sha256": authorization["sha256"],
                "scientific_projection": projection,
                "scientific_projection_sha256": _canonical_sha256(projection),
                "checkpoint_observation_sha256": _canonical_sha256(
                    {"observation": observation}
                ),
                "estimator_ledger_sha256": _sha256_file(ledger_path),
                "checkpoint_file_sha256": _sha256_file(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "witness": {
                    "final_state": dict(final_state),
                    "accepted_transition": dict(transitions[0]),
                },
                "scientific_execution_performed": True,
                "diagnostic_only": True,
                "campaign_cell_progress_credited": False,
                "scientific_artifacts_retained": False,
                "paper_evidence_adoption_authorized": False,
            }
        )


def _guarded_parity_branch(
    *,
    variant: str,
    activation_dir: Path,
    authorization: Mapping[str, Any],
    runtime_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capacity = _capacity(runtime_dir)
    if capacity["status"] != "passed":
        raise LocalStrongSectorError(
            "Capacity blocked parity branch: "
            + ", ".join(capacity["blockers"])
        )
    scratch = Path(
        tempfile.mkdtemp(prefix=f"paper-i-strong5-parity-{variant}.")
    )
    stdout_path = scratch / "stdout.json"
    stderr_path = scratch / "stderr.log"
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
            **LOCAL_NUMERICAL_THREAD_ENVIRONMENT,
            "TMPDIR": scratch.as_posix(),
            PARITY_CHILD_TOKEN_ENV: _expected_parity_child_token(
                authorization, variant
            ),
        }
    )
    command = [
        sys.executable,
        "-B",
        ADAPTER_PATH.as_posix(),
        "--activation-dir",
        activation_dir.as_posix(),
        "--parity-branch",
        variant,
    ]
    started = time.monotonic()
    process: subprocess.Popen[Any] | None = None
    peak_rss = 0
    minimum_available: int | None = None
    minimum_free_disk: int | None = None
    stop_reason: str | None = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=False,
            )
        child = psutil.Process(process.pid)
        while process.poll() is None:
            rss = _aggregate_rss(child)
            available = int(psutil.virtual_memory().available)
            free_disk = int(shutil.disk_usage(scratch).free)
            peak_rss = max(peak_rss, rss)
            minimum_available = (
                available
                if minimum_available is None
                else min(minimum_available, available)
            )
            minimum_free_disk = (
                free_disk
                if minimum_free_disk is None
                else min(minimum_free_disk, free_disk)
            )
            if rss > RSS_LIMIT_BYTES:
                stop_reason = "rss_limit_exceeded"
            elif available < AVAILABLE_MEMORY_FLOOR_BYTES:
                stop_reason = "available_memory_floor_crossed"
            elif free_disk < RUNTIME_FREE_DISK_FLOOR_BYTES:
                stop_reason = "runtime_free_disk_floor_crossed"
            elif time.monotonic() - started > PARITY_TIMEOUT_SECONDS:
                stop_reason = "parity_timeout_exceeded"
            if stop_reason is not None:
                _terminate_process_tree(process)
                break
            time.sleep(0.5)
        returncode = int(process.returncode or 0)
        if stop_reason is not None:
            raise LocalStrongSectorError(
                f"Parity branch guard stopped {variant}: {stop_reason}"
            )
        if returncode != 0:
            error = stderr_path.read_text(encoding="utf-8", errors="replace")
            raise LocalStrongSectorError(
                f"Parity branch failed {variant}={returncode}: {error[-4000:]}"
            )
        child_payload = _validate_inline_digested(
            json.loads(stdout_path.read_text(encoding="utf-8")),
            label=f"{variant} parity child payload",
        )
        projection = child_payload.get("scientific_projection")
        if (
            not isinstance(projection, dict)
            or child_payload.get("variant") != variant
            or child_payload.get("execution_id") != TARGET_EXECUTION_IDS[0]
            or child_payload.get("controller_rounds_completed")
            != PARITY_MAXIMUM_ROUNDS
            or child_payload.get("scientific_projection_sha256")
            != _canonical_sha256(projection)
        ):
            raise LocalStrongSectorError("Parity child projection drifted.")
        branch = _digested(
            {
                "schema": PARITY_BRANCH_SCHEMA,
                "status": "passed_guarded_authorized_diagnostic_branch",
                "variant": variant,
                "execution_id": TARGET_EXECUTION_IDS[0],
                "controller_rounds_completed": PARITY_MAXIMUM_ROUNDS,
                "job_spec_sha256": child_payload["job_spec_sha256"],
                "protocol_sha256": child_payload["protocol_sha256"],
                "route_contract_sha256": child_payload[
                    "route_contract_sha256"
                ],
                "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "current_checkpoint_source_sha256": child_payload[
                    "current_checkpoint_source_sha256"
                ],
                "execution_plan_sha256": child_payload[
                    "execution_plan_sha256"
                ],
                "execution_authorization_sha256": authorization["sha256"],
                "child_payload_sha256": child_payload["sha256"],
                "scientific_projection_sha256": child_payload[
                    "scientific_projection_sha256"
                ],
                "checkpoint_observation_sha256": child_payload[
                    "checkpoint_observation_sha256"
                ],
                "estimator_ledger_sha256": child_payload[
                    "estimator_ledger_sha256"
                ],
                "checkpoint_file_sha256": child_payload[
                    "checkpoint_file_sha256"
                ],
                "checkpoint_size_bytes": child_payload[
                    "checkpoint_size_bytes"
                ],
                "witness": child_payload["witness"],
                "resource_guard": {
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_rss_bytes": peak_rss,
                    "rss_limit_bytes": RSS_LIMIT_BYTES,
                    "minimum_available_memory_bytes": minimum_available,
                    "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
                    "minimum_free_disk_bytes": minimum_free_disk,
                    "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
                    "guard_stop_reason": None,
                    "child_returncode": returncode,
                },
                "scientific_execution_performed": True,
                "diagnostic_only": True,
                "campaign_cell_progress_credited": False,
                "scientific_artifacts_retained": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        return projection, branch
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _validate_scientific_parity_canary(
    *,
    activation_dir: Path,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    parity = _load_digested(
        activation_dir / "scientific_parity_canary.json",
        label="scientific parity canary",
    )
    baseline = _verify_binding(
        activation_dir,
        parity.get("sealed_baseline_branch"),
        expected_path="parity_sealed_baseline_branch.json",
        label="sealed parity branch",
        canonical=True,
    )
    candidate = _verify_binding(
        activation_dir,
        parity.get("operational_candidate_branch"),
        expected_path="parity_operational_candidate_branch.json",
        label="candidate parity branch",
        canonical=True,
    )
    assert baseline is not None and candidate is not None
    expected_sources = {
        "sealed_baseline": SEALED_CURRENT_CHECKPOINT_SHA256,
        "operational_candidate": CANDIDATE_CURRENT_CHECKPOINT_SHA256,
    }
    for branch, variant in zip((baseline, candidate), PARITY_VARIANTS):
        if (
            branch.get("schema") != PARITY_BRANCH_SCHEMA
            or branch.get("status")
            != "passed_guarded_authorized_diagnostic_branch"
            or branch.get("variant") != variant
            or branch.get("execution_id") != TARGET_EXECUTION_IDS[0]
            or branch.get("controller_rounds_completed")
            != PARITY_MAXIMUM_ROUNDS
            or branch.get("job_spec_sha256") != PARITY_JOB_SPEC_SHA256
            or branch.get("protocol_sha256") != PARITY_PROTOCOL_SHA256
            or branch.get("route_contract_sha256") != PARITY_ROUTE_SHA256
            or branch.get("current_checkpoint_source_sha256")
            != expected_sources[variant]
            or branch.get("execution_plan_sha256") != plan.get("sha256")
            or branch.get("execution_authorization_sha256")
            != authorization.get("sha256")
            or branch.get("resource_guard", {}).get("guard_stop_reason")
            is not None
            or branch.get("resource_guard", {}).get("child_returncode") != 0
            or branch.get("scientific_execution_performed") is not True
            or branch.get("diagnostic_only") is not True
            or branch.get("campaign_cell_progress_credited") is not False
            or branch.get("scientific_artifacts_retained") is not False
            or branch.get("paper_evidence_adoption_authorized") is not False
        ):
            raise LocalStrongSectorError("Scientific parity branch drifted.")
    baseline_witness = baseline.get("witness", {})
    final_state = baseline_witness.get("final_state", {})
    expected = _digested(
        {
            "schema": PARITY_SCHEMA,
            "status": "passed_exact_scientific_parity",
            "completed_at_utc": parity.get("completed_at_utc"),
            "execution_id": TARGET_EXECUTION_IDS[0],
            "maximum_controller_rounds": PARITY_MAXIMUM_ROUNDS,
            "parity_canary_spec_sha256": plan[
                "scientific_parity_canary_spec"
            ]["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "execution_authorization_sha256": authorization["sha256"],
            "sealed_baseline_branch": _binding(
                activation_dir / "parity_sealed_baseline_branch.json",
                root=activation_dir,
                canonical=True,
            ),
            "operational_candidate_branch": _binding(
                activation_dir / "parity_operational_candidate_branch.json",
                root=activation_dir,
                canonical=True,
            ),
            "scientific_projection_sha256": baseline.get(
                "scientific_projection_sha256"
            ),
            "estimator_ledger_sha256": baseline.get(
                "estimator_ledger_sha256"
            ),
            "one_round_energy": final_state.get("energy"),
            "exact_canonical_projection_equal": True,
            "estimator_ledger_equal": True,
            "scientific_execution_performed": True,
            "diagnostic_only": True,
            "campaign_cell_progress_credited": False,
            "scientific_artifacts_retained": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    if (
        baseline.get("scientific_projection_sha256")
        != candidate.get("scientific_projection_sha256")
        or baseline.get("estimator_ledger_sha256")
        != candidate.get("estimator_ledger_sha256")
        or baseline.get("witness") != candidate.get("witness")
        or final_state.get("energy") != PARITY_BASELINE_ENERGY
        or parity != expected
    ):
        raise LocalStrongSectorError("Exact scientific parity did not close.")
    return parity


def _materialize_scientific_parity_canary(
    *,
    activation_dir: Path,
    runtime_dir: Path,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_projection, baseline = _guarded_parity_branch(
        variant="sealed_baseline",
        activation_dir=activation_dir,
        authorization=authorization,
        runtime_dir=runtime_dir,
    )
    baseline_path = activation_dir / "parity_sealed_baseline_branch.json"
    _write_json_exclusive(baseline_path, baseline)
    candidate_projection, candidate = _guarded_parity_branch(
        variant="operational_candidate",
        activation_dir=activation_dir,
        authorization=authorization,
        runtime_dir=runtime_dir,
    )
    candidate_path = activation_dir / "parity_operational_candidate_branch.json"
    _write_json_exclusive(candidate_path, candidate)
    projection_difference = _first_canonical_difference(
        baseline_projection, candidate_projection
    )
    ledger_equal = (
        baseline["estimator_ledger_sha256"]
        == candidate["estimator_ledger_sha256"]
    )
    witness_difference = _first_canonical_difference(
        baseline["witness"], candidate["witness"]
    )
    if projection_difference is not None or not ledger_equal or witness_difference:
        raise LocalStrongSectorError(
            "Operational checkpoint repair changed one-round parity: "
            f"projection={projection_difference}; "
            f"estimator_ledger_equal={ledger_equal}; "
            f"witness={witness_difference}."
        )
    energy = baseline["witness"]["final_state"]["energy"]
    if energy != PARITY_BASELINE_ENERGY:
        raise LocalStrongSectorError("Parity baseline energy witness drifted.")
    parity = _digested(
        {
            "schema": PARITY_SCHEMA,
            "status": "passed_exact_scientific_parity",
            "completed_at_utc": _utc_now(),
            "execution_id": TARGET_EXECUTION_IDS[0],
            "maximum_controller_rounds": PARITY_MAXIMUM_ROUNDS,
            "parity_canary_spec_sha256": plan[
                "scientific_parity_canary_spec"
            ]["sha256"],
            "execution_plan_sha256": plan["sha256"],
            "execution_authorization_sha256": authorization["sha256"],
            "sealed_baseline_branch": _binding(
                baseline_path, root=activation_dir, canonical=True
            ),
            "operational_candidate_branch": _binding(
                candidate_path, root=activation_dir, canonical=True
            ),
            "scientific_projection_sha256": baseline[
                "scientific_projection_sha256"
            ],
            "estimator_ledger_sha256": baseline[
                "estimator_ledger_sha256"
            ],
            "one_round_energy": energy,
            "exact_canonical_projection_equal": True,
            "estimator_ledger_equal": True,
            "scientific_execution_performed": True,
            "diagnostic_only": True,
            "campaign_cell_progress_credited": False,
            "scientific_artifacts_retained": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    _write_json_exclusive(
        activation_dir / "scientific_parity_canary.json", parity
    )
    return _validate_scientific_parity_canary(
        activation_dir=activation_dir,
        plan=plan,
        authorization=authorization,
    )


def authorize_activation(
    *,
    planning_dir: Path,
    activation_dir: Path,
    runtime_dir: Path,
    authorization_basis: str,
) -> dict[str, Any]:
    if not authorization_basis.strip():
        raise LocalStrongSectorError("Explicit local authorization basis is absent.")
    worker = _load_worker()
    manifest, _rows = _closed_inputs(worker)
    planning, plan, _preflight, remote_hold = _validate_planning(
        planning_dir, manifest=manifest, require_fresh_hold=True
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
        for name in (
            "execution_plan.json",
            "host_preflight.json",
            "remote_hold_receipt.json",
            "planning_manifest.json",
        ):
            source = planning_dir / name
            destination = temporary / name
            shutil.copy2(source, destination)
            if _binding(source, root=planning_dir, canonical=True) != _binding(
                destination, root=temporary, canonical=True
            ):
                raise LocalStrongSectorError(
                    f"Planning copy drifted during authorization: {name}"
                )
        copied_planning, copied_plan, _copied_preflight, copied_hold = (
            _validate_planning(
                temporary,
                manifest=manifest,
                require_fresh_hold=True,
                reject_authority_files=False,
            )
        )
        if (
            copied_planning != planning
            or copied_plan != plan
            or copied_hold != remote_hold
        ):
            raise LocalStrongSectorError("Copied planning contract drifted.")
        authorization = _digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "status": "authorized_local_execution",
                "created_at_utc": _utc_now(),
                "authorization_kind": "explicit_user_local_execution_authority",
                "authorization_basis": authorization_basis.strip(),
                "user_scope_confirmation": (
                    "five unfinished Paper-I strong-Holstein-sector jobs"
                ),
                "planning_manifest_sha256": planning["sha256"],
                "execution_plan": planning["execution_plan"],
                "execution_plan_sha256": plan["sha256"],
                "execution_ids": list(TARGET_EXECUTION_IDS),
                "cluster_id": 9647385,
                "held_procs": sorted(TARGET_PROCS),
                "remote_hold_receipt_sha256": remote_hold["sha256"],
                "execution_target": LOCAL_EXECUTION_TARGET,
                "maximum_concurrency": MAXIMUM_CONCURRENCY,
                "fresh_start_only": True,
                "scientific_parity_canary_authorized": True,
                "parity_canary_spec_sha256": plan[
                    "scientific_parity_canary_spec"
                ]["sha256"],
                "parity_canary_execution_id": TARGET_EXECUTION_IDS[0],
                "parity_canary_variants": list(PARITY_VARIANTS),
                "parity_canary_maximum_rounds": PARITY_MAXIMUM_ROUNDS,
                "parity_canary_campaign_progress_credited": False,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "remote_release_authorized": False,
            }
        )
        authorization_path = temporary / "execution_authorization.json"
        _write_json_exclusive(authorization_path, authorization)
        parity = _materialize_scientific_parity_canary(
            activation_dir=temporary,
            runtime_dir=runtime_dir,
            plan=plan,
            authorization=authorization,
        )
        activation = _digested(
            {
                "schema": ACTIVATION_SCHEMA,
                "status": "passed_local_activation_authorized",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": _sha256_file(ADAPTER_PATH),
                "planning_manifest": _binding(
                    temporary / "planning_manifest.json",
                    root=temporary,
                    canonical=True,
                ),
                "execution_plan": planning["execution_plan"],
                "execution_authorization": _binding(
                    authorization_path, root=temporary, canonical=True
                ),
                "scientific_parity_canary": _binding(
                    temporary / "scientific_parity_canary.json",
                    root=temporary,
                    canonical=True,
                ),
                "host_preflight": planning["host_preflight"],
                "remote_hold_receipt": planning["remote_hold_receipt"],
                "execution_ids": list(TARGET_EXECUTION_IDS),
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json_exclusive(temporary / "activation_manifest.json", activation)
        os.rename(temporary, activation_dir)
        return activation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_activation(
    activation_dir: Path,
    *,
    manifest: Mapping[str, Any],
    require_fresh_hold: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    planning, plan, preflight, remote_hold = _validate_planning(
        activation_dir,
        manifest=manifest,
        require_fresh_hold=require_fresh_hold,
        reject_authority_files=False,
    )
    activation = _load_digested(
        activation_dir / "activation_manifest.json",
        label="local activation manifest",
    )
    verified_planning = _verify_binding(
        activation_dir,
        activation.get("planning_manifest"),
        expected_path="planning_manifest.json",
        label="local planning manifest",
        canonical=True,
    )
    authorization = _verify_binding(
        activation_dir,
        activation.get("execution_authorization"),
        expected_path="execution_authorization.json",
        label="execution authorization",
        canonical=True,
    )
    assert verified_planning is not None and authorization is not None
    parity = _verify_binding(
        activation_dir,
        activation.get("scientific_parity_canary"),
        expected_path="scientific_parity_canary.json",
        label="scientific parity canary",
        canonical=True,
    )
    assert parity is not None
    parity = _validate_scientific_parity_canary(
        activation_dir=activation_dir,
        plan=plan,
        authorization=authorization,
    )
    if (
        verified_planning != planning
        or activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status") != "passed_local_activation_authorized"
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or activation.get("local_adapter_sha256") != _sha256_file(ADAPTER_PATH)
        or activation.get("execution_plan") != planning.get("execution_plan")
        or activation.get("host_preflight") != planning.get("host_preflight")
        or activation.get("remote_hold_receipt")
        != planning.get("remote_hold_receipt")
        or activation.get("execution_ids") != list(TARGET_EXECUTION_IDS)
        or activation.get("execution_target") != LOCAL_EXECUTION_TARGET
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("planning_manifest_sha256")
        != planning.get("sha256")
        or authorization.get("execution_plan_sha256") != plan.get("sha256")
        or authorization.get("execution_ids") != list(TARGET_EXECUTION_IDS)
        or authorization.get("remote_hold_receipt_sha256")
        != remote_hold.get("sha256")
        or authorization.get("scientific_parity_canary_authorized")
        is not True
        or authorization.get("parity_canary_spec_sha256")
        != plan["scientific_parity_canary_spec"]["sha256"]
        or authorization.get("parity_canary_execution_id")
        != TARGET_EXECUTION_IDS[0]
        or authorization.get("parity_canary_variants")
        != list(PARITY_VARIANTS)
        or authorization.get("parity_canary_maximum_rounds")
        != PARITY_MAXIMUM_ROUNDS
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("execution_target") != LOCAL_EXECUTION_TARGET
        or preflight.get("scientific_execution_performed") is not False
        or activation.get("scientific_parity_canary", {}).get(
            "canonical_sha256"
        )
        != parity.get("sha256")
    ):
        raise LocalStrongSectorError("Local activation contract drifted.")
    return activation, plan, authorization, remote_hold


def _runtime_manifest(activation: Mapping[str, Any]) -> dict[str, Any]:
    return _digested(
        {
            "schema": RUNTIME_SCHEMA,
            "status": "authorized_pending_serial_cells",
            "adapter_path": ADAPTER_PATH.relative_to(REPO_ROOT).as_posix(),
            "adapter_sha256": _sha256_file(ADAPTER_PATH),
            "source_package_id": PACKAGE_DIR.name,
            "package_manifest_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "activation_manifest_sha256": activation["sha256"],
            "scientific_parity_canary_sha256": activation[
                "scientific_parity_canary"
            ]["canonical_sha256"],
            "execution_ids": list(TARGET_EXECUTION_IDS),
            "execution_target": LOCAL_EXECUTION_TARGET,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "checkpoint_keep_history_tail": CHECKPOINT_KEEP_HISTORY_TAIL,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
            "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _ensure_runtime(
    runtime_dir: Path, *, activation: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _runtime_manifest(activation)
    names = (
        "runs",
        "worker_receipts",
        "guard_receipts",
        "logs",
        "status",
        "in_progress",
        "quarantine",
        "tmp",
    )
    if runtime_dir.exists() or runtime_dir.is_symlink():
        if not runtime_dir.is_dir() or runtime_dir.is_symlink():
            raise LocalStrongSectorError("Runtime destination is unsafe.")
        observed = _load_digested(
            runtime_dir / "runtime_manifest.json", label="runtime manifest"
        )
        if observed != expected:
            raise LocalStrongSectorError("Runtime manifest drifted.")
        for name in names:
            path = runtime_dir / name
            if not path.is_dir() or path.is_symlink():
                raise LocalStrongSectorError(f"Runtime directory is incomplete: {name}")
        return observed
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{runtime_dir.name}.build-", dir=runtime_dir.parent
        )
    )
    try:
        for name in names:
            (temporary / name).mkdir()
        _write_json_exclusive(temporary / "runtime_manifest.json", expected)
        os.rename(temporary, runtime_dir)
        return expected
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _expected_child_token(runtime: Mapping[str, Any], execution_id: str) -> str:
    return hashlib.sha256(
        f"{runtime['sha256']}:{execution_id}:local-child-v1".encode("utf-8")
    ).hexdigest()


def _stream_result_and_summary(worker: Any, staging: Path, result: Any) -> None:
    result_payload = result.to_dict()
    if not isinstance(result_payload, Mapping):
        raise LocalStrongSectorError("Scientific result projection is malformed.")
    _write_json_streaming(staging / "result/result.json", result_payload)
    del result_payload
    summary = result.run.paper_i_summary
    if summary is None or not callable(getattr(summary, "to_dict", None)):
        raise LocalStrongSectorError("Paper-I summary is absent.")
    summary_payload = summary.to_dict()
    if not isinstance(summary_payload, Mapping):
        raise LocalStrongSectorError("Paper-I summary projection is malformed.")
    _write_json_streaming(staging / "summary/summary.json", summary_payload)


def _compact_checkpoint_observation_factory(original: Any) -> Any:
    """Return a constructor that changes only checkpoint tail retention."""

    def compact_checkpoint_observation(*args: Any, **kwargs: Any) -> Any:
        kwargs["keep_history_tail"] = CHECKPOINT_KEEP_HISTORY_TAIL
        return original(*args, **kwargs)

    return compact_checkpoint_observation


def _source_overlay_receipt(
    *,
    source_root: Path,
    execution_id: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    applied = _copy_overlays(source_root)
    return _digested(
        {
            "schema": SOURCE_OVERLAY_SCHEMA,
            "status": "passed_operational_source_overlay",
            "execution_id": execution_id,
            "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "execution_plan_sha256": plan["sha256"],
            "overlay_files": applied,
            "checkpoint_keep_history_tail": CHECKPOINT_KEEP_HISTORY_TAIL,
            "scientific_protocol_settings_changed": False,
            "route_contracts_changed": False,
            "observation_serialization_only": True,
        }
    )


def _run_local_cell(
    *,
    execution_id: str,
    activation_dir: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    if execution_id not in TARGET_BY_ID:
        raise LocalStrongSectorError("Internal child is outside the five-cell scope.")
    worker = _load_worker()
    manifest, rows = _closed_inputs(worker)
    activation, plan, authorization, _remote_hold = _validate_activation(
        activation_dir, manifest=manifest
    )
    runtime = _ensure_runtime(runtime_dir, activation=activation)
    if os.environ.get(LOCAL_CHILD_TOKEN_ENV) != _expected_child_token(
        runtime, execution_id
    ):
        raise LocalStrongSectorError(
            "Cell execution is available only to the guarded serial supervisor."
        )
    row = next(row for row in rows if row["execution_id"] == execution_id)
    output_dir = runtime_dir / "runs" / execution_id
    external_receipt = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    attempt = runtime_dir / "in_progress" / execution_id
    if any(
        path.exists() or path.is_symlink()
        for path in (output_dir, external_receipt, attempt)
    ):
        raise LocalStrongSectorError(
            f"Refusing to overwrite a prior cell attempt: {execution_id}"
        )
    attempt.mkdir()
    source_root = attempt / "source"
    staging = attempt / "cell_output"
    try:
        worker._extract_source(
            manifest=manifest,
            source_locks=row["source_locks"],
            destination=source_root,
        )
        overlay = _source_overlay_receipt(
            source_root=source_root,
            execution_id=execution_id,
            plan=plan,
        )
        staging.mkdir()
        original_cwd = Path.cwd()
        os.chdir(source_root)
        try:
            worker._activate_source_root(source_root)
            protocol, problem = worker._load_protocol(
                job=row["job"],
                payload=row["protocol"],
                source_locks=row["source_locks"],
            )
            if (
                protocol.sha256 != row["job"]["protocol_sha256"]
                or protocol.route_contract["sha256"]
                != row["job"]["route_contract_sha256"]
            ):
                raise LocalStrongSectorError("Materialized protocol drifted.")
            from pipelines.static_adapt import adapt_pipeline
            import pipelines.static_adapt.sr_snake as sr_snake

            adapt_pipeline._ai_log = lambda *args, **kwargs: None
            original_checkpoint_observation = sr_snake.CheckpointObservation

            sr_snake.CheckpointObservation = (
                _compact_checkpoint_observation_factory(
                    original_checkpoint_observation
                )
            )
            try:
                result, rounds = worker._execute(
                    protocol=protocol,
                    problem=problem,
                    staging=staging,
                    maximum_rounds=TARGET_HORIZON,
                )
            finally:
                sr_snake.CheckpointObservation = original_checkpoint_observation
        finally:
            os.chdir(original_cwd)
        if rounds != TARGET_HORIZON:
            raise LocalStrongSectorError(
                f"Cell stopped at k={rounds}, not k={TARGET_HORIZON}: "
                f"{execution_id}"
            )
        _stream_result_and_summary(worker, staging, result)
        overlay_path = staging / "provenance/local_source_overlay_receipt.json"
        _write_json_exclusive(overlay_path, overlay)
        expected_paths = worker._expected_artifact_paths(row["job"])
        artifact_digest_cache: dict[str, tuple[str, int]] = {}
        output_payloads: dict[str, dict[str, Any]] = {}
        for role, relative in expected_paths.items():
            if role == "execution_manifest":
                continue
            path = staging / relative
            digest = _sha256_file(path)
            size = path.stat().st_size
            artifact_digest_cache[relative] = (digest, size)
            output_payloads[role] = {
                "path": str(row["job"]["expected_run_artifacts"][role]["path"]),
                "sha256": digest,
                "size_bytes": size,
            }
        if set(output_payloads) != set(expected_paths).difference(
            {"execution_manifest"}
        ):
            raise LocalStrongSectorError("Required scientific artifact is absent.")
        execution_manifest = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                    "execution_manifest_v1"
                ),
                "status": "passed",
                "package_id": manifest["package_id"],
                "campaign_id": manifest["campaign_id"],
                "execution_id": execution_id,
                "job_spec_sha256": row["job"]["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_plan_sha256": plan["sha256"],
                "scientific_parity_canary_sha256": activation[
                    "scientific_parity_canary"
                ]["canonical_sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "target_horizon": TARGET_HORIZON,
                "comparator_policy": row["job"]["comparator_policy"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "source_checkpoint_consumed": False,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "base_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "source_overlay_receipt_sha256": overlay["sha256"],
                "checkpoint_keep_history_tail": CHECKPOINT_KEEP_HISTORY_TAIL,
                "scientific_protocol_settings_changed": False,
                "paper_evidence_adoption_authorized": False,
                "output_payloads": output_payloads,
            }
        )
        _write_json_exclusive(
            staging / expected_paths["execution_manifest"], execution_manifest
        )
        if any(
            not (staging / relative).is_file()
            for relative in expected_paths.values()
        ):
            raise LocalStrongSectorError("Expected-artifact closure is incomplete.")
        artifacts: list[dict[str, Any]] = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(staging).as_posix()
            cached = artifact_digest_cache.get(relative)
            digest, size = (
                cached
                if cached is not None
                else (_sha256_file(path), path.stat().st_size)
            )
            artifacts.append(
                {
                    "path": (
                        PurePosixPath("runs") / execution_id / relative
                    ).as_posix(),
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
        receipt = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
                    "worker_receipt_v1"
                ),
                "status": "passed",
                "package_id": manifest["package_id"],
                "campaign_id": manifest["campaign_id"],
                "execution_id": execution_id,
                "job_spec_sha256": row["job"]["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_plan_sha256": plan["sha256"],
                "scientific_parity_canary_sha256": activation[
                    "scientific_parity_canary"
                ]["canonical_sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "source_overlay_receipt_sha256": overlay["sha256"],
                "checkpoint_keep_history_tail": CHECKPOINT_KEEP_HISTORY_TAIL,
                "artifacts": artifacts,
            }
        )
        internal_receipt = staging / "provenance/local_worker_receipt.json"
        _write_json_exclusive(internal_receipt, receipt)
        os.rename(staging, output_dir)
        os.link(
            output_dir / "provenance/local_worker_receipt.json",
            external_receipt,
        )
        shutil.rmtree(source_root)
        attempt.rmdir()
        return receipt
    except BaseException as exc:
        failure = _digested(
            {
                "schema": "paper_i_page12_strong_sector5_local_failure_v1",
                "status": "preserved_failed_or_interrupted_attempt",
                "created_at_utc": _utc_now(),
                "execution_id": execution_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "staging_preserved": attempt.is_dir(),
                "retry_execution_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        if attempt.is_dir():
            failure_path = attempt / "failure_receipt.json"
            if not failure_path.exists() and not failure_path.is_symlink():
                _write_json_exclusive(failure_path, failure)
        raise


def _verify_receipt_artifacts(
    runtime_dir: Path, receipt: Mapping[str, Any]
) -> None:
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise LocalStrongSectorError("Worker artifact inventory is absent.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise LocalStrongSectorError("Worker artifact inventory is malformed.")
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise LocalStrongSectorError("Worker artifact path is unsafe.")
        path = runtime_dir / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or _sha256_file(path) != row.get("sha256")
        ):
            raise LocalStrongSectorError(
                f"Worker artifact binding drifted: {relative}"
            )


def _closed_cell(runtime_dir: Path, execution_id: str) -> bool:
    run_root = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    guard_path = runtime_dir / "guard_receipts" / f"{execution_id}.json"
    in_progress = runtime_dir / "in_progress" / execution_id
    quarantined = runtime_dir / "quarantine" / execution_id
    if any(
        path.exists() or path.is_symlink()
        for path in (in_progress, quarantined)
    ):
        raise LocalStrongSectorError(
            f"Preserved or quarantined state blocks closure: {execution_id}"
        )
    if not any(
        path.exists() or path.is_symlink()
        for path in (run_root, receipt_path, guard_path)
    ):
        return False
    if (
        not run_root.is_dir()
        or run_root.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
        or not guard_path.is_file()
        or guard_path.is_symlink()
    ):
        raise LocalStrongSectorError(
            f"Incomplete published output requires inspection: {execution_id}"
        )
    manifest = _load_digested(
        run_root / "execution_manifest.json",
        label=f"execution manifest {execution_id}",
    )
    receipt = _load_digested(
        receipt_path, label=f"worker receipt {execution_id}"
    )
    guard = _load_digested(
        guard_path, label=f"guard receipt {execution_id}"
    )
    internal_receipt = run_root / "provenance/local_worker_receipt.json"
    runtime = _load_digested(
        runtime_dir / "runtime_manifest.json", label="runtime manifest"
    )
    if (
        manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("target_horizon") != TARGET_HORIZON
        or int(manifest.get("controller_rounds_completed", 0))
        != TARGET_HORIZON
        or manifest.get("checkpoint_keep_history_tail")
        != CHECKPOINT_KEEP_HISTORY_TAIL
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or manifest.get("scientific_parity_canary_sha256")
        != runtime.get("scientific_parity_canary_sha256")
        or receipt.get("scientific_parity_canary_sha256")
        != runtime.get("scientific_parity_canary_sha256")
        or guard.get("schema") != GUARD_SCHEMA
        or guard.get("status") != "passed"
        or guard.get("execution_id") != execution_id
        or guard.get("child_returncode") != 0
        or guard.get("guard_stop_reason") is not None
        or guard.get("execution_manifest_sha256") != manifest.get("sha256")
        or not internal_receipt.is_file()
        or _sha256_file(internal_receipt) != _sha256_file(receipt_path)
    ):
        raise LocalStrongSectorError(f"Completed cell closure drifted: {execution_id}")
    _verify_receipt_artifacts(runtime_dir, receipt)
    return True


def _terminal_cell_binding(
    runtime_dir: Path, execution_id: str
) -> dict[str, Any]:
    if not _closed_cell(runtime_dir, execution_id):
        raise LocalStrongSectorError(
            f"Terminal closure is absent: {execution_id}"
        )
    return {
        "execution_id": execution_id,
        "execution_manifest": _binding(
            runtime_dir / "runs" / execution_id / "execution_manifest.json",
            root=runtime_dir,
            canonical=True,
        ),
        "worker_receipt": _binding(
            runtime_dir / "worker_receipts" / f"{execution_id}.json",
            root=runtime_dir,
            canonical=True,
        ),
        "guard_receipt": _binding(
            runtime_dir / "guard_receipts" / f"{execution_id}.json",
            root=runtime_dir,
            canonical=True,
        ),
    }


def _terminal_receipt_payload(
    *,
    activation_dir: Path,
    runtime_dir: Path,
    activation: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    runtime: Mapping[str, Any],
    completed_at_utc: str,
) -> dict[str, Any]:
    return _digested(
        {
            "schema": TERMINAL_SCHEMA,
            "status": "passed_all_five_cells_immutable_closure",
            "completed_at_utc": completed_at_utc,
            "execution_ids": list(TARGET_EXECUTION_IDS),
            "completed_execution_ids": list(TARGET_EXECUTION_IDS),
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "authority": {
                "planning_manifest": _binding(
                    activation_dir / "planning_manifest.json",
                    root=activation_dir,
                    canonical=True,
                ),
                "execution_plan": _binding(
                    activation_dir / "execution_plan.json",
                    root=activation_dir,
                    canonical=True,
                ),
                "execution_authorization": _binding(
                    activation_dir / "execution_authorization.json",
                    root=activation_dir,
                    canonical=True,
                ),
                "activation_manifest": _binding(
                    activation_dir / "activation_manifest.json",
                    root=activation_dir,
                    canonical=True,
                ),
                "scientific_parity_canary": _binding(
                    activation_dir / "scientific_parity_canary.json",
                    root=activation_dir,
                    canonical=True,
                ),
                "planning_manifest_sha256": _load_digested(
                    activation_dir / "planning_manifest.json",
                    label="terminal planning manifest",
                )["sha256"],
                "execution_plan_sha256": plan["sha256"],
                "execution_authorization_sha256": authorization["sha256"],
                "activation_manifest_sha256": activation["sha256"],
            },
            "runtime_manifest": _binding(
                runtime_dir / "runtime_manifest.json",
                root=runtime_dir,
                canonical=True,
            ),
            "runtime_manifest_sha256": runtime["sha256"],
            "cells": [
                _terminal_cell_binding(runtime_dir, execution_id)
                for execution_id in TARGET_EXECUTION_IDS
            ],
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _validate_terminal_receipt(
    *,
    activation_dir: Path,
    runtime_dir: Path,
    activation: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    path = runtime_dir / "terminal_receipt.json"
    terminal = _load_digested(path, label="local terminal receipt")
    _utc_datetime(
        terminal.get("completed_at_utc"), label="terminal completed_at_utc"
    )
    expected = _terminal_receipt_payload(
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
        activation=activation,
        plan=plan,
        authorization=authorization,
        runtime=runtime,
        completed_at_utc=str(terminal.get("completed_at_utc")),
    )
    if terminal != expected:
        raise LocalStrongSectorError("Immutable terminal receipt drifted.")
    return terminal


def _write_terminal_receipt(
    *,
    activation_dir: Path,
    runtime_dir: Path,
    activation: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = _terminal_receipt_payload(
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
        activation=activation,
        plan=plan,
        authorization=authorization,
        runtime=runtime,
        completed_at_utc=_utc_now(),
    )
    _write_json_atomic_noreplace(
        runtime_dir / "terminal_receipt.json", terminal
    )
    return _validate_terminal_receipt(
        activation_dir=activation_dir,
        runtime_dir=runtime_dir,
        activation=activation,
        plan=plan,
        authorization=authorization,
        runtime=runtime,
    )


def _scientific_overlap() -> list[str]:
    own_pid = os.getpid()
    matches: list[str] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            command = " ".join(process.info.get("cmdline") or ())
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            continue
        if pid == own_pid or not command:
            continue
        if (
            (ADAPTER_PATH.as_posix() in command and "--run-cell" in command)
            or ("run_local_page16_insertion_comparators_20260812.py" in command
                and "--run-cell" in command)
            or ("run_cell.py" in command and "--run" in command)
            or "local_runner.py run-cell" in command
        ):
            matches.append(f"{pid} {command}")
    return matches


def inert_preflight(
    *, planning_dir: Path, activation_dir: Path, runtime_dir: Path
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_inputs(worker)
    planning_status = "absent"
    external_planning: dict[str, Any] | None = None
    if planning_dir.exists() or planning_dir.is_symlink():
        external_planning, _plan, _host, _hold = _validate_planning(
            planning_dir, manifest=manifest, require_fresh_hold=False
        )
        planning_status = "validated_not_authorized"
    activation_status = "absent"
    if activation_dir.exists() or activation_dir.is_symlink():
        _validate_activation(
            activation_dir, manifest=manifest, require_fresh_hold=False
        )
        copied_planning = _load_digested(
            activation_dir / "planning_manifest.json",
            label="activated planning manifest",
        )
        if (
            external_planning is not None
            and copied_planning != external_planning
        ):
            raise LocalStrongSectorError(
                "Authorized activation does not match the inert planning record."
            )
        activation_status = "validated_authorized"
    capacity = _capacity(runtime_dir)
    preflights = [_sealed_worker_preflight(row) for row in rows]
    overlay_preflight = _isolated_overlay_preflight(TARGET_EXECUTION_IDS[0])
    runtime_status = "absent"
    completed: list[str] = []
    preserved_attempts: list[str] = []
    if runtime_dir.exists() or runtime_dir.is_symlink():
        if activation_status != "validated_authorized":
            raise LocalStrongSectorError("Runtime exists without valid activation.")
        activation, _plan, _authority, _remote_hold = _validate_activation(
            activation_dir, manifest=manifest, require_fresh_hold=False
        )
        _ensure_runtime(runtime_dir, activation=activation)
        completed = [
            execution_id
            for execution_id in TARGET_EXECUTION_IDS
            if _closed_cell(runtime_dir, execution_id)
        ]
        preserved_attempts = [
            execution_id
            for execution_id in TARGET_EXECUTION_IDS
            if (runtime_dir / "in_progress" / execution_id).exists()
            or (runtime_dir / "in_progress" / execution_id).is_symlink()
        ]
        runtime_status = "validated"
    overlap = _scientific_overlap()
    run_ready = (
        activation_status == "validated_authorized"
        and capacity["status"] == "passed"
        and not preserved_attempts
        and not overlap
    )
    return _digested(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed_inert_preflight",
            "package_manifest_sha256": manifest["sha256"],
            "local_adapter_sha256": _sha256_file(ADAPTER_PATH),
            "execution_ids": list(TARGET_EXECUTION_IDS),
            "sealed_worker_preflights": preflights,
            "operational_overlay_preflight": overlay_preflight,
            "planning_status": planning_status,
            "activation_status": activation_status,
            "runtime_status": runtime_status,
            "completed_execution_ids": completed,
            "preserved_in_progress_attempts": preserved_attempts,
            "overlapping_scientific_commands": overlap,
            "capacity": capacity,
            "run_ready": run_ready,
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def _aggregate_rss(process: psutil.Process) -> int:
    total = 0
    candidates = [process]
    try:
        candidates.extend(process.children(recursive=True))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    for candidate in candidates:
        try:
            total += int(candidate.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return total


def _checkpoint_depth(path: Path) -> int | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            stream.seek(max(0, size - CHECKPOINT_TAIL_SCAN_BYTES))
            tail = stream.read().decode("utf-8", errors="ignore")
    except OSError:
        return None
    matches = re.findall(r'"history_count"\s*:\s*(\d+)', tail)
    return None if not matches else int(matches[-1])


def _status_payload(
    *,
    runtime: Mapping[str, Any],
    status: str,
    completed: Sequence[str],
    current_execution_id: str | None,
    child_pid: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
    terminal_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "status": status,
        "updated_at_utc": _utc_now(),
        "runtime_manifest_sha256": runtime["sha256"],
        "execution_ids": list(TARGET_EXECUTION_IDS),
        "completed_execution_ids": list(completed),
        "current_execution_id": current_execution_id,
        "child_pid": child_pid,
        "maximum_concurrency": MAXIMUM_CONCURRENCY,
    }
    if metrics is not None:
        value["metrics"] = dict(metrics)
    if failure is not None:
        value["failure"] = dict(failure)
    if terminal_receipt_sha256 is not None:
        value["terminal_receipt_sha256"] = terminal_receipt_sha256
    return _digested(value)


def _emit_event(event: str, **fields: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at_utc": _utc_now(), **fields},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        descendants = []
    for descendant in reversed(descendants):
        try:
            descendant.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        for descendant in reversed(descendants):
            try:
                descendant.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        process.kill()
        process.wait()


def _guarded_cell(
    *,
    execution_id: str,
    activation_dir: Path,
    runtime_dir: Path,
    runtime: Mapping[str, Any],
    completed: Sequence[str],
) -> dict[str, Any]:
    capacity = _capacity(runtime_dir)
    if capacity["status"] != "passed":
        raise LocalStrongSectorError(
            "Capacity blocked cell launch: " + ", ".join(capacity["blockers"])
        )
    stdout_path = runtime_dir / "logs" / f"{execution_id}.out"
    stderr_path = runtime_dir / "logs" / f"{execution_id}.err"
    guard_path = runtime_dir / "guard_receipts" / f"{execution_id}.json"
    if any(path.exists() or path.is_symlink() for path in (stdout_path, stderr_path, guard_path)):
        raise LocalStrongSectorError(f"Refusing to overwrite cell logs: {execution_id}")
    command = [
        sys.executable,
        "-B",
        ADAPTER_PATH.as_posix(),
        "--activation-dir",
        activation_dir.as_posix(),
        "--runtime-dir",
        runtime_dir.as_posix(),
        "--run-cell",
        execution_id,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
            **LOCAL_NUMERICAL_THREAD_ENVIRONMENT,
            "TMPDIR": (runtime_dir / "tmp").as_posix(),
            LOCAL_CHILD_TOKEN_ENV: _expected_child_token(runtime, execution_id),
        }
    )
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            # The private remote-runner owns a dedicated process group for
            # this fixed job.  Inheriting it guarantees that a remote-runner
            # SIGKILL cannot orphan an unguarded scientific child.
            start_new_session=False,
        )
    child = psutil.Process(process.pid)
    peak_rss = 0
    minimum_available: int | None = None
    minimum_free_disk: int | None = None
    checkpoint_depth: int | None = None
    last_reported_depth: int | None = None
    stop_reason: str | None = None
    last_status_write = 0.0
    status_path = runtime_dir / "status/campaign.json"
    _emit_event(
        "cell_started",
        execution_id=execution_id,
        pid=process.pid,
        completed=list(completed),
    )
    try:
        while process.poll() is None:
            rss = _aggregate_rss(child)
            memory = psutil.virtual_memory()
            disk = shutil.disk_usage(runtime_dir)
            available = int(memory.available)
            free_disk = int(disk.free)
            peak_rss = max(peak_rss, rss)
            minimum_available = (
                available
                if minimum_available is None
                else min(minimum_available, available)
            )
            minimum_free_disk = (
                free_disk
                if minimum_free_disk is None
                else min(minimum_free_disk, free_disk)
            )
            observed_depth = _checkpoint_depth(
                runtime_dir
                / "in_progress"
                / execution_id
                / "cell_output/checkpoints/current.json"
            )
            if observed_depth is not None:
                checkpoint_depth = max(checkpoint_depth or 0, observed_depth)
                if checkpoint_depth != last_reported_depth:
                    _emit_event(
                        "checkpoint_persisted",
                        execution_id=execution_id,
                        checkpoint_depth=checkpoint_depth,
                        checkpoint_path=(
                            runtime_dir
                            / "in_progress"
                            / execution_id
                            / "cell_output/checkpoints/current.json"
                        ).as_posix(),
                    )
                    last_reported_depth = checkpoint_depth
            if rss > RSS_LIMIT_BYTES:
                stop_reason = "rss_limit_exceeded"
            elif available < AVAILABLE_MEMORY_FLOOR_BYTES:
                stop_reason = "available_memory_floor_crossed"
            elif free_disk < RUNTIME_FREE_DISK_FLOOR_BYTES:
                stop_reason = "runtime_free_disk_floor_crossed"
            now = time.monotonic()
            metrics = {
                "elapsed_seconds": now - started,
                "current_rss_bytes": rss,
                "peak_rss_bytes": peak_rss,
                "available_memory_bytes": available,
                "minimum_available_memory_bytes": minimum_available,
                "free_disk_bytes": free_disk,
                "minimum_free_disk_bytes": minimum_free_disk,
                "checkpoint_depth": checkpoint_depth,
            }
            if now - last_status_write >= STATUS_WRITE_SECONDS:
                _write_json_atomic(
                    status_path,
                    _status_payload(
                        runtime=runtime,
                        status="running_serial_cell",
                        completed=completed,
                        current_execution_id=execution_id,
                        child_pid=process.pid,
                        metrics=metrics,
                    ),
                )
                last_status_write = now
            if stop_reason is not None:
                _terminate_process_tree(process)
                break
            time.sleep(GUARD_POLL_SECONDS)
    except BaseException:
        _terminate_process_tree(process)
        raise
    returncode = int(process.returncode or 0)
    elapsed = time.monotonic() - started
    execution_manifest_sha256 = None
    manifest_path = runtime_dir / "runs" / execution_id / "execution_manifest.json"
    if returncode == 0 and manifest_path.is_file():
        execution_manifest_sha256 = _load_digested(
            manifest_path, label=f"execution manifest {execution_id}"
        )["sha256"]
    guard = _digested(
        {
            "schema": GUARD_SCHEMA,
            "status": "passed" if returncode == 0 and stop_reason is None else "stopped",
            "created_at_utc": _utc_now(),
            "execution_id": execution_id,
            "child_pid": process.pid,
            "child_returncode": returncode,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "minimum_available_memory_bytes": minimum_available,
            "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
            "minimum_free_disk_bytes": minimum_free_disk,
            "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
            "maximum_checkpoint_depth_observed": checkpoint_depth,
            "guard_stop_reason": stop_reason,
            "execution_manifest_sha256": execution_manifest_sha256,
        }
    )
    _write_json_exclusive(guard_path, guard)
    if stop_reason is not None:
        raise LocalStrongSectorError(
            f"Guard stopped {execution_id}: {stop_reason}"
        )
    if returncode != 0:
        raise LocalStrongSectorError(
            f"Cell child failed: {execution_id}={returncode}; see {stderr_path}"
        )
    if not _closed_cell(runtime_dir, execution_id):
        raise LocalStrongSectorError(f"Cell did not close: {execution_id}")
    _emit_event(
        "cell_completed",
        execution_id=execution_id,
        elapsed_seconds=elapsed,
        peak_rss_bytes=peak_rss,
        maximum_checkpoint_depth_observed=checkpoint_depth,
    )
    return guard


def run_campaign(
    *, activation_dir: Path, runtime_dir: Path
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, _rows = _closed_inputs(worker)
    activation, plan, authorization, _remote_hold = _validate_activation(
        activation_dir, manifest=manifest, require_fresh_hold=False
    )
    runtime = _ensure_runtime(runtime_dir, activation=activation)
    lock_path = runtime_dir / "campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LocalStrongSectorError("Another strong-sector supervisor is active.") from exc
        terminal_path = runtime_dir / "terminal_receipt.json"
        if terminal_path.exists() or terminal_path.is_symlink():
            if not terminal_path.is_file() or terminal_path.is_symlink():
                raise LocalStrongSectorError("Terminal receipt path is unsafe.")
            return _validate_terminal_receipt(
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
                activation=activation,
                plan=plan,
                authorization=authorization,
                runtime=runtime,
            )
        fresh_activation = _validate_activation(
            activation_dir, manifest=manifest, require_fresh_hold=True
        )
        if fresh_activation[:3] != (activation, plan, authorization):
            raise LocalStrongSectorError("Activation drifted before launch.")
        overlap = _scientific_overlap()
        if overlap:
            raise LocalStrongSectorError(
                "Another local scientific worker is active: " + " | ".join(overlap)
            )
        completed = [
            execution_id
            for execution_id in TARGET_EXECUTION_IDS
            if _closed_cell(runtime_dir, execution_id)
        ]
        for execution_id in TARGET_EXECUTION_IDS:
            if execution_id in completed:
                continue
            attempt = runtime_dir / "in_progress" / execution_id
            if attempt.exists() or attempt.is_symlink():
                raise LocalStrongSectorError(
                    f"Preserved attempt requires inspection: {execution_id}"
                )
            try:
                _guarded_cell(
                    execution_id=execution_id,
                    activation_dir=activation_dir,
                    runtime_dir=runtime_dir,
                    runtime=runtime,
                    completed=completed,
                )
            except BaseException as exc:
                failed = _status_payload(
                    runtime=runtime,
                    status="failed_or_guard_stopped",
                    completed=completed,
                    current_execution_id=execution_id,
                    failure={
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                _write_json_atomic(runtime_dir / "status/campaign.json", failed)
                raise
            completed.append(execution_id)
            _write_json_atomic(
                runtime_dir / "status/campaign.json",
                _status_payload(
                    runtime=runtime,
                    status="cell_passed_pending_remaining",
                    completed=completed,
                    current_execution_id=None,
                ),
            )
        terminal = _write_terminal_receipt(
            activation_dir=activation_dir,
            runtime_dir=runtime_dir,
            activation=activation,
            plan=plan,
            authorization=authorization,
            runtime=runtime,
        )
        final = _status_payload(
            runtime=runtime,
            status="passed_all_five_cells",
            completed=completed,
            current_execution_id=None,
            terminal_receipt_sha256=terminal["sha256"],
        )
        _write_json_atomic(runtime_dir / "status/campaign.json", final)
        _emit_event(
            "campaign_completed",
            completed=completed,
            terminal_receipt_sha256=terminal["sha256"],
        )
        return terminal


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guarded local Paper-I Page-12 strong-sector five-cell runner"
    )
    parser.add_argument(
        "--planning-dir", type=Path, default=DEFAULT_PLANNING_DIR
    )
    parser.add_argument(
        "--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR
    )
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument(
        "--remote-hold-receipt",
        type=Path,
        default=DEFAULT_REMOTE_HOLD_RECEIPT,
    )
    parser.add_argument("--authorization-basis")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--authorize", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-campaign", action="store_true")
    mode.add_argument("--run-cell", help=argparse.SUPPRESS)
    mode.add_argument(
        "--parity-branch", choices=PARITY_VARIANTS, help=argparse.SUPPRESS
    )
    mode.add_argument("--overlay-preflight", help=argparse.SUPPRESS)
    args = parser.parse_args()
    planning_dir = args.planning_dir.resolve()
    activation_dir = args.activation_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    try:
        if args.prepare:
            if args.authorization_basis is not None:
                raise LocalStrongSectorError(
                    "--authorization-basis is valid only with --authorize."
                )
            payload = prepare_planning(
                planning_dir=planning_dir,
                runtime_dir=runtime_dir,
                remote_hold_receipt=args.remote_hold_receipt.resolve(),
            )
        elif args.authorize:
            if args.authorization_basis is None:
                raise LocalStrongSectorError(
                    "--authorize requires --authorization-basis."
                )
            payload = authorize_activation(
                planning_dir=planning_dir,
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
                authorization_basis=args.authorization_basis,
            )
        elif args.preflight:
            if args.authorization_basis is not None:
                raise LocalStrongSectorError(
                    "--authorization-basis is valid only with --authorize."
                )
            payload = inert_preflight(
                planning_dir=planning_dir,
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
            )
        elif args.run_campaign:
            if args.authorization_basis is not None:
                raise LocalStrongSectorError(
                    "--authorization-basis is valid only with --authorize."
                )
            payload = run_campaign(
                activation_dir=activation_dir, runtime_dir=runtime_dir
            )
        elif args.run_cell is not None:
            if args.authorization_basis is not None:
                raise LocalStrongSectorError(
                    "--authorization-basis is valid only with --authorize."
                )
            payload = _run_local_cell(
                execution_id=str(args.run_cell),
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
            )
        elif args.parity_branch is not None:
            if args.authorization_basis is not None:
                raise LocalStrongSectorError(
                    "--authorization-basis is valid only with --authorize."
                )
            payload = _run_parity_branch(
                variant=str(args.parity_branch),
                activation_dir=activation_dir,
            )
        else:
            payload = _overlay_protocol_preflight(
                execution_id=str(args.overlay_preflight)
            )
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        psutil.Error,
        LocalStrongSectorError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(_canonical_json_bytes(payload).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
