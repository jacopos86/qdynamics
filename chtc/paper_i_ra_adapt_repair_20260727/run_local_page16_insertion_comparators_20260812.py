#!/usr/bin/env python3
"""Run the ten unfinished Page-16 insertion comparators locally to k=30.

This is a local-only execution adapter around the sealed CHTC package.  It
does not alter the sealed package, reuse CHTC authorizations, submit work, or
derive/execute a round-50 continuation.  The ten unfinished cells are exposed
as five explicit two-regime waves.  Each completed cell receives a
``paper_i_effective_plateau_v1`` round-30 gate receipt that records whether
the effective prefix was observed before the cap or is right-censored at it.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ADAPTER_PATH = Path(__file__).resolve()
REPAIR_ROOT = ADAPTER_PATH.parent
REPO_ROOT = ADAPTER_PATH.parents[2]
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_weak50_strong30_"
    "20260812_v1_chtc"
)
DEFAULT_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_insertion_comparators_k30_"
    "20260812_v2_local_activation"
)
DEFAULT_RUNTIME_DIR = REPO_ROOT / (
    "output/local_runs/"
    "paper_i_page16_insertion_comparators_k30_20260812_v2"
)

PACKAGE_MANIFEST_FILE_SHA256 = (
    "6830624199a2bddaecf5fdea9a9a27584a4f6b9011023165ee816af51b373f34"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "7d8a00196f25f8dd90c08b626dd8e4083868ca93e68aa883568f29b63d4bf0b1"
)
SOURCE_ARCHIVE_SHA256 = (
    "95b3ea575a4590961b6a57337eb1c58ef3ba3855d9d342b179657973c129ef26"
)
LOCAL_TARGET_HORIZON = 30
CONTINUATION_TARGET_HORIZON = 50
WAVE_SIZE = 2
# Comparable completed Page-16 workers used about 9.8 GiB RSS each.  The
# active Mac has 16 GiB physical memory, so the two members of each requested
# two-regime wave must execute serially on this host.
MAX_CONCURRENCY = 1
MIN_MEMORY_PRESSURE_FREE_PERCENT = 20
MIN_FREE_DISK_BYTES_BY_WAVE = {
    # Prior byte-closed local k=50 Page-16 macro cells occupy at most 2.27 GiB
    # each.  Reserve 4 GiB per k=30 cell for published output plus another
    # 4 GiB per live child for isolated source/staging and write headroom.
    wave_number: 16 * 1024**3 for wave_number in range(1, 6)
}
LOCAL_CHILD_TOKEN_ENV = "PAPER_I_PAGE16_LOCAL_K30_WAVE_SUPERVISOR"
LOCAL_EXECUTION_TARGET = "local_mac_two_regime_waves_v2"

LOCAL_REQUEST_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_activation_request_v2"
)
LOCAL_PREFLIGHT_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_host_preflight_v2"
)
LOCAL_AUTHORIZATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_authorization_v2"
)
LOCAL_ACTIVATION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_activation_manifest_v2"
)
LOCAL_RUNTIME_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_runtime_manifest_v2"
)
LOCAL_STATUS_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_wave_status_v2"
)
LOCAL_EXECUTION_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_execution_manifest_v2"
)
LOCAL_WORKER_RECEIPT_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_worker_receipt_v2"
)
LOCAL_QUARANTINE_RECEIPT_SCHEMA = (
    "paper_i_page16_insertion_comparator_local_k30_quarantine_receipt_v1"
)
PLATEAU_GATE_SCHEMA = (
    "paper_i_page16_insertion_comparator_k30_effective_plateau_gate_v2"
)

EXECUTION_PREFIX = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes"
)
ROUTE_TOKEN = (
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes"
)
REGIMES: tuple[tuple[str, int], ...] = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
POLICIES = ("always_commutation_reduced", "append_only")


class LocalRunError(RuntimeError):
    """Raised when a local execution or evidence contract is not closed."""


def _execution_id(regime: str, nph: int, policy: str) -> str:
    return (
        f"{EXECUTION_PREFIX}__{regime}__nph{int(nph)}__"
        f"{ROUTE_TOKEN}_{policy}"
    )


PACKAGE_EXECUTION_IDS = tuple(
    _execution_id(regime, nph, policy)
    for policy in POLICIES
    for regime, nph in REGIMES
)
COMPLETED_ALWAYS_OPEN_IDS = (
    _execution_id("weak_weak", 3, "always_commutation_reduced"),
    _execution_id("intermediate_weak", 3, "always_commutation_reduced"),
)
WAVES: tuple[tuple[str, str], ...] = (
    (
        _execution_id("weak_weak", 3, "append_only"),
        _execution_id("intermediate_weak", 3, "append_only"),
    ),
    (
        _execution_id("strong_weak_u8", 3, "always_commutation_reduced"),
        _execution_id("weak_strong", 7, "always_commutation_reduced"),
    ),
    (
        _execution_id("strong_weak_u8", 3, "append_only"),
        _execution_id("weak_strong", 7, "append_only"),
    ),
    (
        _execution_id(
            "intermediate_strong", 7, "always_commutation_reduced"
        ),
        _execution_id("strong_strong_u8", 7, "always_commutation_reduced"),
    ),
    (
        _execution_id("intermediate_strong", 7, "append_only"),
        _execution_id("strong_strong_u8", 7, "append_only"),
    ),
)
TARGET_EXECUTION_IDS = tuple(row for wave in WAVES for row in wave)
WAVE_BY_EXECUTION_ID = {
    execution_id: wave_number
    for wave_number, wave in enumerate(WAVES, start=1)
    for execution_id in wave
}
REGIME_BY_EXECUTION_ID = {
    _execution_id(regime, nph, policy): regime
    for regime, nph in REGIMES
    for policy in POLICIES
}
NPH_BY_EXECUTION_ID = {
    _execution_id(regime, nph, policy): nph
    for regime, nph in REGIMES
    for policy in POLICIES
}


_WORKER: Any | None = None


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
        "paper_i_page16_insertion_comparator_sealed_worker",
        PACKAGE_DIR / "run_cell.py",
    )
    if spec is None or spec.loader is None:
        raise LocalRunError("Unable to load the sealed Page-16 worker.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _WORKER = module
    return module


def _write_json_exclusive(
    worker: Any,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(worker.canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_atomic(
    worker: Any,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise LocalRunError(f"Stale status temporary exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(worker.canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_digested(worker: Any, path: Path, *, label: str) -> dict[str, Any]:
    payload = worker.load_json(path, label=label)
    worker.verify_self_digest(payload, label=label)
    return payload


def _binding(
    worker: Any,
    path: Path,
    *,
    root: Path,
    canonical: bool,
) -> dict[str, Any]:
    row = {
        "path": path.relative_to(root).as_posix(),
        "sha256": worker.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        row["canonical_sha256"] = _load_digested(
            worker, path, label=path.name
        )["sha256"]
    return row


def _verify_local_binding(
    worker: Any,
    root: Path,
    raw: Any,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("path") != expected_path:
        raise LocalRunError(f"{label} binding path drifted.")
    path = root / expected_path
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(raw.get("size_bytes", -1))
        or worker.sha256_file(path) != raw.get("sha256")
    ):
        raise LocalRunError(f"{label} byte binding drifted.")
    payload = _load_digested(worker, path, label=label)
    if payload["sha256"] != raw.get("canonical_sha256"):
        raise LocalRunError(f"{label} canonical binding drifted.")
    return payload


def _queue_rows(worker: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue_path, _payload = worker._verify_binding(
        manifest.get("queue"), label="queue"
    )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(
        queue_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split("\t")
        if len(fields) != 8:
            raise LocalRunError(f"Malformed sealed queue row {index}.")
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
        absolute_job = PACKAGE_DIR / worker.safe_relative_path(
            job_path, label=f"queue job {index}"
        )
        if worker.sha256_file(absolute_job) != job_file_sha256:
            raise LocalRunError(f"Sealed queue job drifted at row {index}.")
        rows.append(
            {
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
    if tuple(row["execution_id"] for row in rows) != PACKAGE_EXECUTION_IDS:
        raise LocalRunError("Sealed package queue order or identity drifted.")
    return rows


def _closed_package(
    worker: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    if worker.sha256_file(manifest_path) != PACKAGE_MANIFEST_FILE_SHA256:
        raise LocalRunError("Sealed Page-16 package-manifest bytes drifted.")
    manifest = _load_digested(
        worker, manifest_path, label="sealed package manifest"
    )
    source = manifest.get("source_archive")
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("package_id") != PACKAGE_DIR.name
        or manifest.get("status") != "passed_inert_twelve_cells"
        or manifest.get("row_count") != 12
        or tuple(manifest.get("execution_ids", ())) != PACKAGE_EXECUTION_IDS
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise LocalRunError("Sealed Page-16 package identity drifted.")
    source_path, _unused = worker._verify_binding(
        source, label="source archive"
    )
    if worker.sha256_file(source_path) != SOURCE_ARCHIVE_SHA256:
        raise LocalRunError("Sealed Page-16 source archive drifted.")
    worker._verify_binding(
        manifest.get("source_archive_manifest"),
        label="source archive manifest",
        canonical=True,
    )
    controls = manifest.get("control_files")
    if not isinstance(controls, list):
        raise LocalRunError("Sealed control-file closure is absent.")
    for index, row in enumerate(controls):
        worker._verify_binding(row, label=f"control file {index}")
    rows = _queue_rows(worker, manifest)
    by_id = {row["execution_id"]: row for row in rows}
    if (
        set(TARGET_EXECUTION_IDS).intersection(COMPLETED_ALWAYS_OPEN_IDS)
        or set(TARGET_EXECUTION_IDS).union(COMPLETED_ALWAYS_OPEN_IDS)
        != set(PACKAGE_EXECUTION_IDS)
        or len(TARGET_EXECUTION_IDS) != 10
        or any(len(set(wave)) != 2 for wave in WAVES)
    ):
        raise LocalRunError("Local ten-cell wave partition drifted.")
    for execution_id in TARGET_EXECUTION_IDS:
        row = by_id[execution_id]
        job, package, protocol, _locks = worker._load_closed_job(
            PACKAGE_DIR / row["job_path"]
        )
        expected_source_horizon = 50 if NPH_BY_EXECUTION_ID[execution_id] == 3 else 30
        if (
            package.get("sha256") != manifest.get("sha256")
            or job.get("execution_id") != execution_id
            or job.get("regime_id") != REGIME_BY_EXECUTION_ID[execution_id]
            or int(job.get("nph", -1)) != NPH_BY_EXECUTION_ID[execution_id]
            or int(job.get("target_horizon", -1)) != expected_source_horizon
            or protocol.get("sha256") != job.get("protocol_sha256")
            or int(protocol.get("horizon", -1)) != expected_source_horizon
            or protocol.get("request", {})
            .get("execution", {})
            .get("resume", {})
            .get("kind")
            != "fresh_start"
        ):
            raise LocalRunError(f"Sealed job/protocol drifted: {execution_id}")
    return manifest, rows


def _physical_memory_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(completed.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _available_memory_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True
        )
        lines = completed.stdout.splitlines()
        page_size = 4096
        if lines and "page size of" in lines[0]:
            page_size = int(
                lines[0].split("page size of", 1)[1].split("bytes", 1)[0]
            )
        values: dict[str, int] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            values[key.strip()] = int(raw.strip().rstrip("."))
        pages = sum(
            values.get(key, 0)
            for key in (
                "Pages free",
                "Pages inactive",
                "Pages speculative",
                "Pages purgeable",
            )
        )
        return pages * page_size
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _memory_pressure_free_percent() -> int | None:
    try:
        completed = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        )
        match = re.search(r"free percentage:\s*(\d+)%", completed.stdout)
        return None if match is None else int(match.group(1))
    except (OSError, subprocess.CalledProcessError):
        return None


def _swap_usage() -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise LocalRunError(f"No existing parent for {path}")
        candidate = parent
    return candidate


def _capacity_receipt(
    worker: Any,
    runtime_dir: Path,
    *,
    wave_number: int | None = None,
) -> dict[str, Any]:
    available_memory = _available_memory_bytes()
    pressure_free = _memory_pressure_free_percent()
    free_disk = shutil.disk_usage(_existing_parent(runtime_dir.parent)).free
    required_free_disk = (
        min(MIN_FREE_DISK_BYTES_BY_WAVE.values())
        if wave_number is None
        else MIN_FREE_DISK_BYTES_BY_WAVE[wave_number]
    )
    blockers: list[str] = []
    if available_memory is None:
        blockers.append("available_memory_unavailable")
    elif available_memory <= 0:
        blockers.append("available_memory_nonpositive")
    if pressure_free is None:
        blockers.append("memory_pressure_unavailable")
    elif pressure_free < MIN_MEMORY_PRESSURE_FREE_PERCENT:
        blockers.append("memory_pressure_free_percentage_below_guard")
    if free_disk < required_free_disk:
        blockers.append("free_disk_below_guard")
    return worker.digested(
        {
            "schema": "paper_i_page16_local_k30_capacity_receipt_v2",
            "status": "passed" if not blockers else "blocked",
            "physical_memory_bytes": _physical_memory_bytes(),
            "available_or_reclaimable_memory_bytes": available_memory,
            "memory_pressure_free_percent": pressure_free,
            "swap_usage": _swap_usage(),
            "free_disk_bytes": free_disk,
            "available_memory_is_observed_not_a_fixed_launch_gate": True,
            "required_memory_pressure_free_percent": (
                MIN_MEMORY_PRESSURE_FREE_PERCENT
            ),
            "wave": wave_number,
            "required_free_disk_bytes": required_free_disk,
            "disk_guard_basis": (
                "two_prior_local_page16_k50_outputs_at_4_gib_each_plus_"
                "two_live_source_staging_reserves_at_4_gib_each_v1"
            ),
            "maximum_local_concurrency": MAX_CONCURRENCY,
            "scheduler_resource_envelopes_are_provenance_not_local_"
            "reservations": True,
            "nph3_scheduler_envelope": {
                "request_cpus": 4,
                "request_memory_mb": 32768,
                "request_disk_mb": 61440,
                "max_runtime_seconds": 259200,
            },
            "nph7_scheduler_envelope": {
                "request_cpus": 4,
                "request_memory_mb": 49152,
                "request_disk_mb": 81920,
                "max_runtime_seconds": 259200,
            },
            "blockers": blockers,
            "scientific_execution_performed": False,
        }
    )


def _sealed_preflight_rows(
    worker: Any,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row["execution_id"]): row for row in rows}
    receipts: list[dict[str, Any]] = []
    for execution_id in TARGET_EXECUTION_IDS:
        receipt = worker.preflight(
            PACKAGE_DIR / str(by_id[execution_id]["job_path"])
        )
        if (
            receipt.get("status") != "passed"
            or receipt.get("execution_id") != execution_id
            or receipt.get("fresh_start") is not True
            or int(receipt.get("target_horizon", -1))
            not in {LOCAL_TARGET_HORIZON, CONTINUATION_TARGET_HORIZON}
            or receipt.get("scientific_execution_performed") is not False
        ):
            raise LocalRunError(f"Sealed worker preflight drifted: {execution_id}")
        receipts.append(
            {
                "execution_id": execution_id,
                "source_worker_preflight": receipt,
                "source_authorized_horizon": int(receipt["target_horizon"]),
                "local_operational_target_horizon": LOCAL_TARGET_HORIZON,
                "operational_shortening_only": (
                    int(receipt["target_horizon"]) > LOCAL_TARGET_HORIZON
                ),
            }
        )
    return receipts


def inert_preflight(
    *, activation_dir: Path, runtime_dir: Path
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_package(worker)
    preflights = _sealed_preflight_rows(worker, rows)
    capacity = _capacity_receipt(worker, runtime_dir)
    activation_status = "absent"
    if activation_dir.exists() or activation_dir.is_symlink():
        _validate_activation(worker, activation_dir, manifest=manifest)
        activation_status = "validated"
    return worker.digested(
        {
            "schema": LOCAL_PREFLIGHT_SCHEMA,
            "status": "passed_inert_preflight",
            "package_id": manifest["package_id"],
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "local_adapter_sha256": worker.sha256_file(ADAPTER_PATH),
            "target_execution_ids": list(TARGET_EXECUTION_IDS),
            "excluded_completed_execution_ids": list(
                COMPLETED_ALWAYS_OPEN_IDS
            ),
            "waves": [
                {"wave": index, "execution_ids": list(wave)}
                for index, wave in enumerate(WAVES, start=1)
            ],
            "sealed_worker_preflights": preflights,
            "sealed_worker_preflight_count": len(preflights),
            "capacity": capacity,
            "capacity_ready": capacity["status"] == "passed",
            "run_ready": (
                capacity["status"] == "passed"
                and activation_status == "validated"
            ),
            "activation_status": activation_status,
            "execution_target": LOCAL_EXECUTION_TARGET,
            "scientific_execution_performed": False,
            "submission_performed": False,
        }
    )


def prepare_activation(
    *, activation_dir: Path, runtime_dir: Path
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_package(worker)
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
        adapter_sha256 = worker.sha256_file(ADAPTER_PATH)
        request = worker.digested(
            {
                "schema": LOCAL_REQUEST_SCHEMA,
                "status": "authorized_local_execution",
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": adapter_sha256,
                "requested_execution_ids": list(TARGET_EXECUTION_IDS),
                "excluded_completed_execution_ids": list(
                    COMPLETED_ALWAYS_OPEN_IDS
                ),
                "completed_exclusions": {
                    COMPLETED_ALWAYS_OPEN_IDS[0]: "CHTC 9644571.0",
                    COMPLETED_ALWAYS_OPEN_IDS[1]: "CHTC 9647386.0",
                },
                "waves": [
                    {"wave": index, "execution_ids": list(wave)}
                    for index, wave in enumerate(WAVES, start=1)
                ],
                "scope": (
                    "page16_ten_unfinished_cells_local_k30_"
                    "five_two_regime_waves_v2"
                ),
                "authorization_kind": (
                    "explicit_user_local_execution_authority"
                ),
                "explicit_user_authority_recorded": True,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "source_package_execution_target": "chtc",
                "local_operational_target_horizon": LOCAL_TARGET_HORIZON,
                "wave_size": WAVE_SIZE,
                "maximum_concurrency": MAX_CONCURRENCY,
                "host_memory_safe_serialization": True,
                "scientific_protocol_settings_changed": False,
                "weak_protocol_operationally_shortened_50_to_30": True,
                "strong_protocol_horizon_unchanged_at_30": True,
                "round50_continuation_authorized_for_execution": False,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        request_path = temporary / "activation_request.json"
        _write_json_exclusive(worker, request_path, request)

        sealed_preflights = _sealed_preflight_rows(worker, rows)
        host_preflight = worker.digested(
            {
                "schema": LOCAL_PREFLIGHT_SCHEMA,
                "status": "passed_inert_local_host_preflight",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": adapter_sha256,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "sealed_worker_preflights": sealed_preflights,
                "sealed_worker_preflight_count": len(sealed_preflights),
                "capacity_at_preparation": _capacity_receipt(
                    worker, runtime_dir
                ),
                "capacity_recheck_required_before_each_wave": True,
                "scientific_execution_performed": False,
            }
        )
        preflight_path = temporary / "host_preflight.json"
        _write_json_exclusive(worker, preflight_path, host_preflight)
        request_binding = _binding(
            worker, request_path, root=temporary, canonical=True
        )
        preflight_binding = _binding(
            worker, preflight_path, root=temporary, canonical=True
        )

        rows_by_id = {str(row["execution_id"]): row for row in rows}
        authorizations: list[dict[str, Any]] = []
        for execution_id in TARGET_EXECUTION_IDS:
            row = rows_by_id[execution_id]
            job, _package, protocol, _locks = worker._load_closed_job(
                PACKAGE_DIR / str(row["job_path"])
            )
            authority = worker.digested(
                {
                    "schema": LOCAL_AUTHORIZATION_SCHEMA,
                    "status": "authorized_local_cell_execution",
                    "source_package_id": manifest["package_id"],
                    "source_campaign_id": manifest["campaign_id"],
                    "execution_id": execution_id,
                    "regime_id": job["regime_id"],
                    "nph": int(job["nph"]),
                    "comparator_policy": job["comparator_policy"],
                    "wave": WAVE_BY_EXECUTION_ID[execution_id],
                    "job_spec_sha256": job["sha256"],
                    "job_file_sha256": row["job_file_sha256"],
                    "protocol_sha256": protocol["sha256"],
                    "route_contract_sha256": job[
                        "route_contract_sha256"
                    ],
                    "package_manifest_sha256": manifest["sha256"],
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "local_adapter_sha256": adapter_sha256,
                    "source_authorized_horizon": int(job["target_horizon"]),
                    "local_operational_target_horizon": (
                        LOCAL_TARGET_HORIZON
                    ),
                    "fresh_start": True,
                    "activation_request": request_binding,
                    "host_preflight": preflight_binding,
                    "scope": "single_cell_local_k30_execution_only",
                    "authorization_kind": (
                        "explicit_user_local_execution_authority"
                    ),
                    "execution_target": LOCAL_EXECUTION_TARGET,
                    "scientific_protocol_settings_changed": False,
                    "round50_continuation_authorized_for_execution": False,
                    "execution_authorized": True,
                    "submission_authorized": False,
                    "paper_evidence_adoption_authorized": False,
                    "submitted": False,
                }
            )
            authority_path = (
                temporary / "authorizations" / f"{execution_id}.json"
            )
            _write_json_exclusive(worker, authority_path, authority)
            authorizations.append(
                {
                    "execution_id": execution_id,
                    **_binding(
                        worker,
                        authority_path,
                        root=temporary,
                        canonical=True,
                    ),
                }
            )

        activation = worker.digested(
            {
                "schema": LOCAL_ACTIVATION_SCHEMA,
                "status": "passed_local_activation_prepared",
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": adapter_sha256,
                "activation_request": request_binding,
                "host_preflight": preflight_binding,
                "authorizations": authorizations,
                "authorization_count": len(authorizations),
                "execution_ids": list(TARGET_EXECUTION_IDS),
                "excluded_completed_execution_ids": list(
                    COMPLETED_ALWAYS_OPEN_IDS
                ),
                "waves": [list(wave) for wave in WAVES],
                "local_operational_target_horizon": LOCAL_TARGET_HORIZON,
                "wave_size": WAVE_SIZE,
                "maximum_concurrency": MAX_CONCURRENCY,
                "host_memory_safe_serialization": True,
                "execution_target": LOCAL_EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        _write_json_exclusive(
            worker, temporary / "activation_manifest.json", activation
        )
        os.rename(temporary, activation_dir)
        return activation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_activation(
    worker: Any,
    activation_dir: Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if activation_dir.is_symlink() or not activation_dir.is_dir():
        raise LocalRunError("Local activation directory is absent or unsafe.")
    activation = _load_digested(
        worker,
        activation_dir / "activation_manifest.json",
        label="local activation manifest",
    )
    request = _verify_local_binding(
        worker,
        activation_dir,
        activation.get("activation_request"),
        expected_path="activation_request.json",
        label="local activation request",
    )
    preflight = _verify_local_binding(
        worker,
        activation_dir,
        activation.get("host_preflight"),
        expected_path="host_preflight.json",
        label="local host preflight",
    )
    if (
        activation.get("schema") != LOCAL_ACTIVATION_SCHEMA
        or activation.get("status") != "passed_local_activation_prepared"
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or activation.get("local_adapter_sha256")
        != worker.sha256_file(ADAPTER_PATH)
        or activation.get("execution_ids") != list(TARGET_EXECUTION_IDS)
        or activation.get("excluded_completed_execution_ids")
        != list(COMPLETED_ALWAYS_OPEN_IDS)
        or activation.get("waves") != [list(wave) for wave in WAVES]
        or activation.get("authorization_count") != 10
        or activation.get("local_operational_target_horizon") != 30
        or activation.get("wave_size") != WAVE_SIZE
        or activation.get("maximum_concurrency") != MAX_CONCURRENCY
        or activation.get("host_memory_safe_serialization") is not True
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not False
        or activation.get("submitted") is not False
        or request.get("schema") != LOCAL_REQUEST_SCHEMA
        or request.get("requested_execution_ids")
        != list(TARGET_EXECUTION_IDS)
        or request.get("local_adapter_sha256")
        != activation.get("local_adapter_sha256")
        or request.get("execution_target")
        != LOCAL_EXECUTION_TARGET
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not False
        or request.get("round50_continuation_authorized_for_execution")
        is not False
        or preflight.get("schema") != LOCAL_PREFLIGHT_SCHEMA
        or preflight.get("sealed_worker_preflight_count") != 10
        or preflight.get("local_adapter_sha256")
        != activation.get("local_adapter_sha256")
        or preflight.get("scientific_execution_performed") is not False
    ):
        raise LocalRunError("Local activation contract drifted.")
    rows = activation.get("authorizations")
    if not isinstance(rows, list) or len(rows) != 10 or [
        row.get("execution_id") for row in rows if isinstance(row, Mapping)
    ] != list(TARGET_EXECUTION_IDS):
        raise LocalRunError("Local authorization inventory drifted.")
    for row in rows:
        assert isinstance(row, Mapping)
        execution_id = str(row["execution_id"])
        authority = _verify_local_binding(
            worker,
            activation_dir,
            row,
            expected_path=f"authorizations/{execution_id}.json",
            label=f"local authorization {execution_id}",
        )
        if (
            authority.get("schema") != LOCAL_AUTHORIZATION_SCHEMA
            or authority.get("execution_id") != execution_id
            or authority.get("package_manifest_sha256")
            != manifest.get("sha256")
            or authority.get("source_archive_sha256")
            != SOURCE_ARCHIVE_SHA256
            or authority.get("local_adapter_sha256")
            != activation.get("local_adapter_sha256")
            or authority.get("wave") != WAVE_BY_EXECUTION_ID[execution_id]
            or authority.get("local_operational_target_horizon") != 30
            or authority.get("fresh_start") is not True
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not False
            or authority.get("round50_continuation_authorized_for_execution")
            is not False
        ):
            raise LocalRunError(
                f"Local authorization drifted: {execution_id}"
            )
    return activation


def _authorization_for_cell(
    worker: Any,
    activation_dir: Path,
    *,
    activation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    path = activation_dir / "authorizations" / f"{execution_id}.json"
    authority = _load_digested(
        worker, path, label=f"local authorization {execution_id}"
    )
    if (
        authority.get("schema") != LOCAL_AUTHORIZATION_SCHEMA
        or authority.get("execution_id") != execution_id
        or authority.get("job_spec_sha256") != job.get("sha256")
        or authority.get("protocol_sha256") != job.get("protocol_sha256")
        or authority.get("route_contract_sha256")
        != job.get("route_contract_sha256")
        or authority.get("package_manifest_sha256") != manifest.get("sha256")
        or authority.get("local_adapter_sha256")
        != activation.get("local_adapter_sha256")
        or authority.get("local_operational_target_horizon") != 30
        or authority.get("wave") != WAVE_BY_EXECUTION_ID[execution_id]
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not False
        or activation.get("execution_authorized") is not True
    ):
        raise LocalRunError(f"Cell authorization drifted: {execution_id}")
    return authority


def _float_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=1.0e-15
        )
    except (TypeError, ValueError):
        return False


def _write_summary_for_validation(
    worker: Any,
    path: Path,
    typed_summary: Any,
) -> dict[str, Any]:
    """Write one typed summary and return its exact canonical JSON shape."""

    if (
        isinstance(typed_summary, Mapping)
        or getattr(typed_summary, "schema", None) != "paper_i_run_summary_v1"
        or not callable(getattr(typed_summary, "to_dict", None))
    ):
        raise LocalRunError("Paper-I summary must be the typed canonical receipt.")
    projection = typed_summary.to_dict()
    if not isinstance(projection, Mapping):
        raise LocalRunError("Typed Paper-I summary projection is not a mapping.")
    worker._write_json(path, projection)
    written = worker.load_json(path, label="written Paper-I summary")
    expected_bytes = worker.canonical_json_bytes(written) + b"\n"
    if path.read_bytes() != expected_bytes:
        raise LocalRunError("Written Paper-I summary is not canonical JSON.")
    return written


def _quarantine_post_execute_failure(
    worker: Any,
    *,
    staging: Path,
    runtime_dir: Path,
    execution_id: str,
    failure: BaseException,
) -> Path:
    """Atomically preserve staging after science completed but closure failed."""

    if not staging.is_dir() or staging.is_symlink():
        raise LocalRunError(
            f"Post-execute staging is absent or unsafe: {execution_id}"
        )
    quarantine_root = runtime_dir / "quarantine"
    if not quarantine_root.is_dir() or quarantine_root.is_symlink():
        raise LocalRunError("Runtime quarantine directory is absent or unsafe.")
    destination = quarantine_root / execution_id
    temporary = quarantine_root / f".{execution_id}.quarantine-{os.getpid()}"
    if any(path.exists() or path.is_symlink() for path in (destination, temporary)):
        raise LocalRunError(
            f"Refusing to overwrite prior quarantine: {execution_id}"
        )

    # The wave supervisor binds TMPDIR to runtime/in_progress, so this rename is
    # same-filesystem and preserves the only post-execute staging tree before
    # TemporaryDirectory cleanup can erase it.  If a later receipt write fails,
    # the dot-prefixed quarantine remains deliberately recoverable.
    os.rename(staging, temporary)
    preserved = [
        {
            "path": path.relative_to(temporary).as_posix(),
            "sha256": worker.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    receipt = worker.digested(
        {
            "schema": LOCAL_QUARANTINE_RECEIPT_SCHEMA,
            "status": "preserved_post_execute_closure_failure",
            "execution_id": execution_id,
            "local_adapter_sha256": worker.sha256_file(ADAPTER_PATH),
            "failure_type": type(failure).__name__,
            "failure_message": str(failure),
            "scientific_execution_completed": True,
            "scientific_output_published": (
                runtime_dir / "runs" / execution_id
            ).is_dir(),
            "retry_execution_authorized": False,
            "paper_evidence_adoption_authorized": False,
            "preserved_artifacts": preserved,
        }
    )
    _write_json_exclusive(
        worker,
        temporary / "quarantine_receipt.json",
        receipt,
    )
    os.rename(temporary, destination)
    return destination


def _plateau_gate(
    worker: Any,
    *,
    job: Mapping[str, Any],
    summary: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    trace = summary.get("accepted_error_trace")
    plateau = summary.get("effective_plateau")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("horizon_scope") != "deliberately_stopped_prefix"
        or summary.get("available_controller_rounds") != 30
        or not isinstance(trace, list)
        or len(trace) != 30
        or not isinstance(plateau, Mapping)
    ):
        raise LocalRunError(f"Round-30 summary closure drifted: {execution_id}")
    errors: list[float] = []
    for expected_round, row in enumerate(trace, start=1):
        if (
            not isinstance(row, Mapping)
            or row.get("controller_round") != expected_round
        ):
            raise LocalRunError(
                f"Round-30 accepted trace is not contiguous: {execution_id}"
            )
        error = float(row.get("absolute_energy_error"))
        if not math.isfinite(error) or error < 0.0:
            raise LocalRunError(
                f"Round-30 accepted trace error is invalid: {execution_id}"
            )
        errors.append(error)
    best = min(errors)
    threshold = 1.10 * best
    selected_index = next(
        index for index, error in enumerate(errors) if error <= threshold
    )
    selected_round = selected_index + 1
    selected_error = errors[selected_index]
    if (
        plateau.get("policy") != "paper_i_effective_plateau_v1"
        or plateau.get("controller_round") != selected_round
        or plateau.get("available_horizon_controller_rounds") != 30
        or plateau.get("horizon_scope") != "deliberately_stopped_prefix"
        or not _float_equal(plateau.get("absolute_energy_error"), selected_error)
        or not _float_equal(plateau.get("best_observed_error"), best)
        or not _float_equal(plateau.get("selection_threshold"), threshold)
    ):
        raise LocalRunError(
            f"paper_i_effective_plateau_v1 recomputation drifted: {execution_id}"
        )
    selected_at_cap = selected_round == LOCAL_TARGET_HORIZON
    terminal_in_band = errors[-1] <= threshold
    extension_required = (not terminal_in_band) or selected_at_cap
    checkpoint = staging / "checkpoints/current.json"
    ledger_sidecars = sorted(
        checkpoint.parent.glob(
            "current.estimator_call_ledger_checkpoint.*.json"
        )
    )
    verified_sidecars = sorted(
        checkpoint.parent.glob("current.verified_singleton_resume.*.json")
    )
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or len(ledger_sidecars) != 1
        or len(verified_sidecars) != 1
        or any(path.is_symlink() for path in (*ledger_sidecars, *verified_sidecars))
    ):
        raise LocalRunError(
            f"Round-30 resume checkpoint sibling closure drifted: {execution_id}"
        )

    source_horizon = int(job["target_horizon"])
    return worker.digested(
        {
            "schema": PLATEAU_GATE_SCHEMA,
            "status": "passed",
            "execution_id": execution_id,
            "regime_id": job["regime_id"],
            "nph": int(job["nph"]),
            "comparator_policy": job["comparator_policy"],
            "policy": "paper_i_effective_plateau_v1",
            "relative_tolerance": 0.10,
            "available_horizon_controller_rounds": 30,
            "horizon_scope": "deliberately_stopped_prefix",
            "selected_controller_round": selected_round,
            "selected_absolute_energy_error": selected_error,
            "best_observed_absolute_energy_error": best,
            "selection_threshold": threshold,
            "terminal_absolute_energy_error": errors[-1],
            "terminal_in_effective_band": terminal_in_band,
            "selected_at_cap": selected_at_cap,
            "classification": (
                "endpoint_outside_effective_band_at_k30"
                if not terminal_in_band
                else (
                    "right_censored_at_k30"
                    if selected_at_cap
                    else "effective_plateau_observed_within_k30"
                )
            ),
            "extension_decision": (
                "eligible_for_authenticated_resume_to_k50"
                if extension_required
                else "stop_at_k30"
            ),
            "summary_effective_plateau_matches_recomputation": True,
            "accepted_error_trace_canonical_sha256": hashlib.sha256(
                worker.canonical_json_bytes(trace)
            ).hexdigest(),
            "source_authorized_horizon": source_horizon,
            "continuation_target_horizon": CONTINUATION_TARGET_HORIZON,
            "continuation_materialization_requirement": (
                "authenticated_resume_adapter_only"
                if source_horizon >= CONTINUATION_TARGET_HORIZON
                else "new_source_locked_k50_protocol_required"
            ),
            "resume_checkpoint": {
                "path": "checkpoints/current.json",
                "sha256": worker.sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "resume_checkpoint_siblings": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": worker.sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (*ledger_sidecars, *verified_sidecars)
            ],
            "resume_execution_performed": False,
            "round50_protocol_derived": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _run_local_cell(
    *,
    execution_id: str,
    activation_dir: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    if execution_id not in TARGET_EXECUTION_IDS:
        raise LocalRunError("Internal local cell is outside the ten-cell scope.")
    worker = _load_worker()
    manifest, rows = _closed_package(worker)
    activation = _validate_activation(
        worker, activation_dir, manifest=manifest
    )
    if not runtime_dir.is_dir() or runtime_dir.is_symlink():
        raise LocalRunError("Internal cell requires an initialized runtime.")
    runtime_manifest = _ensure_runtime(
        worker, runtime_dir=runtime_dir, activation=activation
    )
    expected_child_token = (
        f"{runtime_manifest['sha256']}:"
        f"wave-{WAVE_BY_EXECUTION_ID[execution_id]}"
    )
    if os.environ.get(LOCAL_CHILD_TOKEN_ENV) != expected_child_token:
        raise LocalRunError(
            "Internal cell execution is available only to its wave supervisor."
        )
    row = next(row for row in rows if row["execution_id"] == execution_id)
    job_path = PACKAGE_DIR / str(row["job_path"])
    job, prepared_manifest, protocol, problem, temporary = worker._prepare(
        job_path
    )
    scientific_execution_completed = False
    try:
        authority = _authorization_for_cell(
            worker,
            activation_dir,
            activation=activation,
            manifest=prepared_manifest,
            job=job,
        )
        output_dir = runtime_dir / "runs" / execution_id
        receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
        external_gate_path = runtime_dir / "plateau_gates" / f"{execution_id}.json"
        if any(
            path.exists() or path.is_symlink()
            for path in (output_dir, receipt_path, external_gate_path)
        ):
            raise LocalRunError(
                f"Refusing to overwrite a prior cell attempt: {execution_id}"
            )
        source_root = Path(temporary.name) / "source"
        staging = Path(temporary.name) / "cell_output"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = worker._execute(
                protocol=protocol,
                problem=problem,
                staging=staging,
                maximum_rounds=LOCAL_TARGET_HORIZON,
            )
            scientific_execution_completed = True
        finally:
            os.chdir(original)
        if rounds != LOCAL_TARGET_HORIZON:
            raise LocalRunError(
                f"Cell stopped at k={rounds}, not k=30: {execution_id}"
            )
        worker._write_json(staging / "result/result.json", result.to_dict())
        if result.run.paper_i_summary is None:
            raise LocalRunError(f"Paper-I summary is absent: {execution_id}")
        summary = _write_summary_for_validation(
            worker,
            staging / "summary/summary.json",
            result.run.paper_i_summary,
        )
        gate = _plateau_gate(
            worker, job=job, summary=summary, staging=staging
        )
        gate_path = staging / "gate/round30_effective_plateau.json"
        worker._write_json(gate_path, gate)

        expected = worker._expected_artifact_paths(job)
        payloads = {
            role: {
                "path": str(job["expected_run_artifacts"][role]["path"]),
                "sha256": worker.sha256_file(staging / relative),
                "size_bytes": (staging / relative).stat().st_size,
            }
            for role, relative in expected.items()
            if role != "execution_manifest"
        }
        payloads["round30_effective_plateau_gate"] = {
            "path": (
                PurePosixPath("runs")
                / execution_id
                / "gate/round30_effective_plateau.json"
            ).as_posix(),
            "sha256": worker.sha256_file(gate_path),
            "size_bytes": gate_path.stat().st_size,
        }
        execution_manifest = worker.digested(
            {
                "schema": LOCAL_EXECUTION_SCHEMA,
                "status": "passed",
                "execution_target": LOCAL_EXECUTION_TARGET,
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "source_package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": activation[
                    "local_adapter_sha256"
                ],
                "execution_id": execution_id,
                "wave": WAVE_BY_EXECUTION_ID[execution_id],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "protocol_sha256": protocol.sha256,
                "route_contract_sha256": protocol.route_contract["sha256"],
                "comparator_policy": job["comparator_policy"],
                "source_authorized_horizon": int(job["target_horizon"]),
                "local_operational_target_horizon": LOCAL_TARGET_HORIZON,
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "source_checkpoint_consumed": False,
                "round50_continuation_executed": False,
                "plateau_gate_sha256": gate["sha256"],
                "output_payloads": payloads,
                "paper_evidence_adoption_authorized": False,
            }
        )
        worker._write_json(
            staging / expected["execution_manifest"], execution_manifest
        )
        worker._publish_staging(staging, output_dir)
        receipt = worker.digested(
            {
                "schema": LOCAL_WORKER_RECEIPT_SCHEMA,
                "status": "passed",
                "execution_target": LOCAL_EXECUTION_TARGET,
                "source_package_id": manifest["package_id"],
                "execution_id": execution_id,
                "wave": WAVE_BY_EXECUTION_ID[execution_id],
                "job_spec_sha256": job["sha256"],
                "authorization_sha256": authority["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "plateau_gate_sha256": gate["sha256"],
                "local_operational_target_horizon": 30,
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "round50_continuation_executed": False,
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
                    if path.is_file()
                ],
            }
        )
        _write_json_exclusive(worker, receipt_path, receipt)
        _write_json_exclusive(worker, external_gate_path, gate)
        return receipt
    except BaseException as exc:
        if scientific_execution_completed and staging.is_dir():
            _quarantine_post_execute_failure(
                worker,
                staging=staging,
                runtime_dir=runtime_dir,
                execution_id=execution_id,
                failure=exc,
            )
        raise
    finally:
        temporary.cleanup()


def _runtime_manifest(
    worker: Any,
    *,
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    return worker.digested(
        {
            "schema": LOCAL_RUNTIME_SCHEMA,
            "status": "authorized_pending_waves",
            "adapter_path": ADAPTER_PATH.relative_to(REPO_ROOT).as_posix(),
            "adapter_sha256": worker.sha256_file(ADAPTER_PATH),
            "source_package_id": PACKAGE_DIR.name,
            "package_manifest_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "activation_manifest_sha256": activation["sha256"],
            "execution_ids": list(TARGET_EXECUTION_IDS),
            "excluded_completed_execution_ids": list(
                COMPLETED_ALWAYS_OPEN_IDS
            ),
            "waves": [
                {"wave": index, "execution_ids": list(wave)}
                for index, wave in enumerate(WAVES, start=1)
            ],
            "local_operational_target_horizon": 30,
            "wave_size": WAVE_SIZE,
            "maximum_concurrency": MAX_CONCURRENCY,
            "host_memory_safe_serialization": True,
            "execution_target": LOCAL_EXECUTION_TARGET,
            "round50_continuation_execution_in_scope": False,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _ensure_runtime(
    worker: Any,
    *,
    runtime_dir: Path,
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _runtime_manifest(worker, activation=activation)
    if runtime_dir.exists() or runtime_dir.is_symlink():
        if runtime_dir.is_symlink() or not runtime_dir.is_dir():
            raise LocalRunError("Local runtime destination is unsafe.")
        observed = _load_digested(
            worker,
            runtime_dir / "runtime_manifest.json",
            label="local runtime manifest",
        )
        if observed != expected:
            raise LocalRunError("Existing local runtime manifest drifted.")
        for name in (
            "runs",
            "worker_receipts",
            "logs",
            "status",
            "plateau_gates",
            "in_progress",
            "quarantine",
        ):
            if not (runtime_dir / name).is_dir():
                raise LocalRunError(f"Runtime directory is incomplete: {name}")
        return observed
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{runtime_dir.name}.build-", dir=runtime_dir.parent
        )
    )
    try:
        for name in (
            "runs",
            "worker_receipts",
            "logs",
            "status",
            "plateau_gates",
            "in_progress",
            "quarantine",
        ):
            (temporary / name).mkdir()
        _write_json_exclusive(
            worker, temporary / "runtime_manifest.json", expected
        )
        os.rename(temporary, runtime_dir)
        return expected
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_receipt_artifacts(
    worker: Any,
    runtime_dir: Path,
    receipt: Mapping[str, Any],
) -> None:
    rows = receipt.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise LocalRunError("Worker artifact inventory is absent.")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise LocalRunError("Worker artifact inventory is malformed.")
        relative = worker.safe_relative_path(
            row.get("path"), label="worker artifact path"
        )
        path = runtime_dir / relative
        name = relative.as_posix()
        if (
            name in seen
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or worker.sha256_file(path) != row.get("sha256")
        ):
            raise LocalRunError(f"Worker artifact binding drifted: {name}")
        seen.add(name)


def _closed_cell(
    worker: Any,
    runtime_dir: Path,
    execution_id: str,
) -> bool:
    run_root = runtime_dir / "runs" / execution_id
    receipt_path = runtime_dir / "worker_receipts" / f"{execution_id}.json"
    external_gate_path = runtime_dir / "plateau_gates" / f"{execution_id}.json"
    paths = (run_root, receipt_path, external_gate_path)
    if not any(path.exists() or path.is_symlink() for path in paths):
        return False
    if not (
        run_root.is_dir()
        and not run_root.is_symlink()
        and receipt_path.is_file()
        and not receipt_path.is_symlink()
        and external_gate_path.is_file()
        and not external_gate_path.is_symlink()
    ):
        raise LocalRunError(
            f"Incomplete published output requires inspection: {execution_id}"
        )
    manifest = _load_digested(
        worker,
        run_root / "execution_manifest.json",
        label=f"execution manifest {execution_id}",
    )
    receipt = _load_digested(
        worker, receipt_path, label=f"worker receipt {execution_id}"
    )
    gate = _load_digested(
        worker, external_gate_path, label=f"plateau gate {execution_id}"
    )
    internal_gate = run_root / "gate/round30_effective_plateau.json"
    if (
        manifest.get("schema") != LOCAL_EXECUTION_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != execution_id
        or manifest.get("local_operational_target_horizon") != 30
        or manifest.get("controller_rounds_completed") != 30
        or manifest.get("fresh_start") is not True
        or manifest.get("round50_continuation_executed") is not False
        or receipt.get("schema") != LOCAL_WORKER_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("execution_id") != execution_id
        or receipt.get("execution_manifest_sha256") != manifest.get("sha256")
        or receipt.get("controller_rounds_completed") != 30
        or gate.get("schema") != PLATEAU_GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("execution_id") != execution_id
        or gate.get("policy") != "paper_i_effective_plateau_v1"
        or gate.get("available_horizon_controller_rounds") != 30
        or gate.get("resume_execution_performed") is not False
        or receipt.get("plateau_gate_sha256") != gate.get("sha256")
        or manifest.get("plateau_gate_sha256") != gate.get("sha256")
        or not internal_gate.is_file()
        or worker.sha256_file(internal_gate)
        != worker.sha256_file(external_gate_path)
    ):
        raise LocalRunError(f"Completed cell closure drifted: {execution_id}")
    _verify_receipt_artifacts(worker, runtime_dir, receipt)
    return True


def _overlapping_scientific_commands() -> list[str]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command=", "-ww"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LocalRunError("Cannot audit local scientific-worker overlap.") from exc
    own_pid = os.getpid()
    matches: list[str] = []
    for raw in output.splitlines():
        text = raw.strip()
        if not text:
            continue
        pid_text, _, command = text.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        if (
            (ADAPTER_PATH.as_posix() in command and "--run-cell" in command)
            or "local_runner.py run-cell" in command
            or ("run_cell.py" in command and " --run " in command)
        ):
            matches.append(text)
    return matches


def _status_payload(
    worker: Any,
    *,
    runtime_manifest: Mapping[str, Any],
    wave_number: int,
    status: str,
    execution_ids: Sequence[str],
    completed: Sequence[str],
    running: Mapping[str, int],
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": LOCAL_STATUS_SCHEMA,
        "status": status,
        "runtime_manifest_sha256": runtime_manifest["sha256"],
        "wave": wave_number,
        "execution_ids": list(execution_ids),
        "completed_execution_ids": list(completed),
        "running_execution_ids": [
            row for row in execution_ids if row in running
        ],
        "running_pids": {
            row: int(running[row]) for row in execution_ids if row in running
        },
        "maximum_concurrency": MAX_CONCURRENCY,
        "local_operational_target_horizon": LOCAL_TARGET_HORIZON,
        "round50_continuation_executed": False,
    }
    if failure is not None:
        payload["failure"] = dict(failure)
    return worker.digested(payload)


def run_wave(
    *,
    wave_number: int,
    activation_dir: Path,
    runtime_dir: Path,
) -> dict[str, Any]:
    if not 1 <= wave_number <= len(WAVES):
        raise LocalRunError("--run-wave must be between 1 and 5.")
    worker = _load_worker()
    manifest, _rows = _closed_package(worker)
    activation = _validate_activation(
        worker, activation_dir, manifest=manifest
    )
    capacity = _capacity_receipt(
        worker, runtime_dir, wave_number=wave_number
    )
    if capacity["status"] != "passed":
        raise LocalRunError(
            "Local capacity gate blocked execution: "
            + ", ".join(capacity["blockers"])
        )
    runtime_manifest = _ensure_runtime(
        worker, runtime_dir=runtime_dir, activation=activation
    )
    lock_path = runtime_dir / "wave_supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(
                lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise LocalRunError("Another Page-16 wave supervisor is active.") from exc
        overlap = _overlapping_scientific_commands()
        if overlap:
            raise LocalRunError(
                "Another local scientific worker is active; refusing overlap: "
                + " | ".join(overlap)
            )
        wave = WAVES[wave_number - 1]
        closed = [
            execution_id
            for execution_id in wave
            if _closed_cell(worker, runtime_dir, execution_id)
        ]
        missing = [row for row in wave if row not in closed]
        status_path = runtime_dir / "status" / f"wave_{wave_number}.json"
        if not missing:
            payload = _status_payload(
                worker,
                runtime_manifest=runtime_manifest,
                wave_number=wave_number,
                status="passed_already_complete",
                execution_ids=wave,
                completed=closed,
                running={},
            )
            _write_json_atomic(worker, status_path, payload)
            return payload

        if MAX_CONCURRENCY != 1:
            raise LocalRunError(
                "This 16-GiB host adapter requires serialized wave members."
            )
        closed_set = set(closed)
        for execution_id in missing:
            stdout_path = runtime_dir / "logs" / f"{execution_id}.out"
            stderr_path = runtime_dir / "logs" / f"{execution_id}.err"
            if any(
                path.exists() or path.is_symlink()
                for path in (stdout_path, stderr_path)
            ):
                raise LocalRunError(
                    f"Refusing to overwrite prior logs: {execution_id}"
                )
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
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "TMPDIR": (runtime_dir / "in_progress").as_posix(),
                    LOCAL_CHILD_TOKEN_ENV: (
                        f"{runtime_manifest['sha256']}:wave-{wave_number}"
                    ),
                }
            )
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=False,
                )
            completed_now = [row for row in wave if row in closed_set]
            try:
                _write_json_atomic(
                    worker,
                    status_path,
                    _status_payload(
                        worker,
                        runtime_manifest=runtime_manifest,
                        wave_number=wave_number,
                        status="running_serialized_wave_member",
                        execution_ids=wave,
                        completed=completed_now,
                        running={execution_id: process.pid},
                    ),
                )
                returncode = process.wait()
            except BaseException as exc:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                interrupted = _status_payload(
                    worker,
                    runtime_manifest=runtime_manifest,
                    wave_number=wave_number,
                    status="interrupted",
                    execution_ids=wave,
                    completed=completed_now,
                    running={},
                    failure={
                        "reason": (
                            "supervisor_keyboard_interrupt"
                            if isinstance(exc, KeyboardInterrupt)
                            else "supervisor_exception"
                        ),
                        "error_type": type(exc).__name__,
                    },
                )
                _write_json_atomic(worker, status_path, interrupted)
                raise
            if returncode != 0:
                failed = _status_payload(
                    worker,
                    runtime_manifest=runtime_manifest,
                    wave_number=wave_number,
                    status="failed",
                    execution_ids=wave,
                    completed=completed_now,
                    running={},
                    failure={
                        "child_returncodes": {execution_id: returncode}
                    },
                )
                _write_json_atomic(worker, status_path, failed)
                raise LocalRunError(
                    f"Wave {wave_number} child failure: "
                    f"{execution_id}={returncode}"
                )
            if not _closed_cell(worker, runtime_dir, execution_id):
                raise LocalRunError(
                    f"Wave {wave_number} did not close {execution_id}."
                )
            closed_set.add(execution_id)
        completed = [
            row for row in wave if _closed_cell(worker, runtime_dir, row)
        ]
        if completed != list(wave):
            raise LocalRunError(
                f"Wave {wave_number} did not publish both closed cells."
            )
        passed = _status_payload(
            worker,
            runtime_manifest=runtime_manifest,
            wave_number=wave_number,
            status="passed",
            execution_ids=wave,
            completed=completed,
            running={},
        )
        _write_json_atomic(worker, status_path, passed)
        return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only Page-16 insertion comparator k=30 wave adapter"
        )
    )
    parser.add_argument(
        "--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-wave", type=int)
    mode.add_argument("--run-cell", help=argparse.SUPPRESS)
    args = parser.parse_args()
    activation_dir = args.activation_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    try:
        if args.prepare:
            payload = prepare_activation(
                activation_dir=activation_dir, runtime_dir=runtime_dir
            )
        elif args.preflight:
            payload = inert_preflight(
                activation_dir=activation_dir, runtime_dir=runtime_dir
            )
        elif args.run_wave is not None:
            payload = run_wave(
                wave_number=args.run_wave,
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
            )
        else:
            payload = _run_local_cell(
                execution_id=str(args.run_cell),
                activation_dir=activation_dir,
                runtime_dir=runtime_dir,
            )
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        LocalRunError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    worker = _load_worker()
    print(worker.canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
