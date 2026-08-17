#!/usr/bin/env python3
"""Materialize and preflight a dormant local Page-12 comparator fallback.

This adapter never submits or executes scientific work.  It closes the exact
sealed twelve-row CHTC package into a separate local-only, pending activation
and reports whether a future serial executor could be enabled.  Such an
executor must remain unavailable until all Page-16 macro k=30 work and every
required k=50 continuation are terminal, a fresh authenticated CHTC snapshot
proves that no Page-12 row or late-materialization factory remains, and the
host has evidence-based memory and disk capacity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ADAPTER_PATH = Path(__file__).resolve()
REPAIR_ROOT = ADAPTER_PATH.parent
REPO_ROOT = ADAPTER_PATH.parents[2]
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
DEFAULT_ACTIVATION_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_"
    "20260812_v1_local_fallback_activation"
)
DEFAULT_MACRO_TERMINAL_RECEIPT = REPAIR_ROOT / (
    "paper_i_ra_adapt_page16_macro_k30_k50_terminal_clearance_20260813.json"
)
MACRO_TERMINAL_PRODUCER_PATH = REPAIR_ROOT / (
    "supervise_local_page16_insertion_comparator_k50_continuations_20260813.py"
)
MACRO_TERMINAL_ADAPTER_PATH = REPAIR_ROOT / (
    "continue_local_page16_insertion_comparators_k30_to_k50_20260813.py"
)
DEFAULT_REMOTE_CLEARANCE = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_"
    "no_remote_overlap_clearance_20260813.json"
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
SEALED_WORKER_SHA256 = (
    "737607f9baa76a1f5fe61791b531457f8e70c2e1e595a93895e1471f5eefe603"
)
EXPECTED_MACRO_TERMINAL_PRODUCER_SHA256 = (
    "0e3a342fa21d925c941a4c3b8e0476c23907ba52d46d459310527a6e0123d761"
)
EXPECTED_MACRO_TERMINAL_ADAPTER_SHA256 = (
    "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
)
TARGET_HORIZON = 50
MAXIMUM_CONCURRENCY = 1
MEMORY_HEADROOM_FACTOR = 1.25
DISK_HEADROOM_FACTOR = 1.25
MIN_MEMORY_PRESSURE_FREE_PERCENT = 20
REMOTE_CLEARANCE_MAX_WINDOW_SECONDS = 15 * 60
MACRO_TERMINAL_REPLAY_TIMEOUT_SECONDS = 30 * 60

LOCAL_REQUEST_SCHEMA = (
    "paper_i_page12_insertion_comparator_local_fallback_request_v1"
)
LOCAL_AUTHORIZATION_SCHEMA = (
    "paper_i_page12_insertion_comparator_local_pending_authorization_v1"
)
LOCAL_ACTIVATION_SCHEMA = (
    "paper_i_page12_insertion_comparator_local_fallback_activation_v1"
)
LOCAL_PREFLIGHT_SCHEMA = (
    "paper_i_page12_insertion_comparator_local_fallback_preflight_v1"
)
MACRO_TERMINAL_SCHEMA = (
    "paper_i_page16_insertion_comparator_macro_k30_k50_terminal_clearance_v1"
)
REMOTE_CLEARANCE_SCHEMA = (
    "paper_i_page12_insertion_comparator_no_remote_overlap_clearance_v1"
)

PAGE16_EXECUTION_PREFIX = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes"
)
PAGE16_ROUTE_TOKEN = (
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes"
)


def _page16_execution_id(regime: str, nph: int, policy: str) -> str:
    return (
        f"{PAGE16_EXECUTION_PREFIX}__{regime}__nph{nph}__"
        f"{PAGE16_ROUTE_TOKEN}_{policy}"
    )


PAGE16_CONDITIONAL_EXECUTION_IDS = (
    _page16_execution_id("weak_weak", 3, "append_only"),
    _page16_execution_id("intermediate_weak", 3, "append_only"),
    _page16_execution_id(
        "weak_strong", 7, "always_commutation_reduced"
    ),
    _page16_execution_id("strong_weak_u8", 3, "append_only"),
    _page16_execution_id("weak_strong", 7, "append_only"),
    _page16_execution_id(
        "intermediate_strong", 7, "always_commutation_reduced"
    ),
    _page16_execution_id(
        "strong_strong_u8", 7, "always_commutation_reduced"
    ),
    _page16_execution_id("intermediate_strong", 7, "append_only"),
    _page16_execution_id("strong_strong_u8", 7, "append_only"),
)
PAGE16_TERMINAL_CHTC_EXECUTION_IDS = (
    _page16_execution_id(
        "weak_weak", 3, "always_commutation_reduced"
    ),
    _page16_execution_id(
        "intermediate_weak", 3, "always_commutation_reduced"
    ),
    _page16_execution_id(
        "strong_weak_u8", 3, "always_commutation_reduced"
    ),
)
MACRO_TERMINAL_PROVENANCE_SHA256_FIELDS = (
    "activation_manifest_sha256",
    "runtime_manifest_sha256",
    "k30_runtime_manifest_sha256",
    "decision_status_sha256",
    "terminal_chtc_status_sha256",
)

EXECUTION_PREFIX = "global_singleton_gradient_phase0_phase23_qiskit_no_lanes"
ROUTE_TOKEN = (
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23"
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


class LocalFallbackError(RuntimeError):
    """Raised when the dormant fallback contract is not closed."""


def _execution_id(regime: str, nph: int, policy: str) -> str:
    return (
        f"{EXECUTION_PREFIX}__{regime}__nph{nph}__"
        f"{ROUTE_TOKEN}_{policy}"
    )


EXECUTION_IDS = tuple(
    _execution_id(regime, nph, policy)
    for policy in POLICIES
    for regime, nph in REGIMES
)

# Runtime/disk/archive values are the closest completed Page-12 plateau cells
# from CHTC cluster 9605157.  Memory guards use the larger of that evidence and
# exact global-singleton nph3 measurements from cluster 9400252.  Cluster
# 9400252 completed the scientific computation and failed only in later
# cross-device publication, so its measured peak RSS remains applicable.
PRIOR_RESOURCE_EVIDENCE: Mapping[str, Mapping[str, Any]] = {
    "weak_weak": {
        "runtime_seconds": 9869,
        "plateau_rss_kib": 12_500_000,
        "global_singleton_peak_memory_mib": 24_113,
        "disk_usage_kib": 3_500_000,
        "compressed_archive_size_bytes": 401_629_564,
        "runtime_cluster_proc": "9605157.0",
        "memory_cluster_proc": "9400252.0",
    },
    "intermediate_weak": {
        "runtime_seconds": 15_833,
        "plateau_rss_kib": 10_000_000,
        "global_singleton_peak_memory_mib": 21_949,
        "disk_usage_kib": 3_750_000,
        "compressed_archive_size_bytes": 410_858_501,
        "runtime_cluster_proc": "9605157.1",
        "memory_cluster_proc": "9400252.1",
    },
    "strong_weak_u8": {
        "runtime_seconds": 11_728,
        "plateau_rss_kib": 10_000_000,
        "global_singleton_peak_memory_mib": 27_847,
        "disk_usage_kib": 3_750_000,
        "compressed_archive_size_bytes": 379_565_071,
        "runtime_cluster_proc": "9605157.2",
        "memory_cluster_proc": "9400252.2",
    },
    "weak_strong": {
        "runtime_seconds": 46_947,
        "plateau_rss_kib": 25_000_000,
        "global_singleton_peak_memory_mib": None,
        "disk_usage_kib": 10_000_000,
        "compressed_archive_size_bytes": 1_099_219_486,
        "runtime_cluster_proc": "9605157.3",
        "memory_cluster_proc": "9605157.3",
    },
    "intermediate_strong": {
        "runtime_seconds": 63_288,
        "plateau_rss_kib": 37_500_000,
        "global_singleton_peak_memory_mib": None,
        "disk_usage_kib": 12_500_000,
        "compressed_archive_size_bytes": 1_169_324_852,
        "runtime_cluster_proc": "9605157.4",
        "memory_cluster_proc": "9605157.4",
    },
    "strong_strong_u8": {
        "runtime_seconds": 89_806,
        "plateau_rss_kib": 75_000_000,
        "global_singleton_peak_memory_mib": None,
        "disk_usage_kib": 17_500_000,
        "compressed_archive_size_bytes": 1_299_721_977,
        "runtime_cluster_proc": "9605157.5",
        "memory_cluster_proc": "9605157.5",
    },
}

_WORKER: Any | None = None


def _load_worker() -> Any:
    global _WORKER
    if _WORKER is not None:
        return _WORKER
    package_text = PACKAGE_DIR.as_posix()
    spec = importlib.util.spec_from_file_location(
        "paper_i_page12_insertion_comparator_sealed_worker",
        PACKAGE_DIR / "run_cell.py",
    )
    if spec is None or spec.loader is None:
        raise LocalFallbackError("Unable to load sealed Page-12 worker.")
    module = importlib.util.module_from_spec(spec)
    prior_path = list(sys.path)
    missing = object()
    prior_contract = sys.modules.pop("package_contract", missing)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != package_text]
        sys.path.insert(0, package_text)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = prior_path
        sys.modules.pop("package_contract", None)
        if prior_contract is not missing:
            sys.modules["package_contract"] = prior_contract
    _WORKER = module
    return module


def _write_json_exclusive(
    worker: Any, path: Path, payload: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(worker.canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_digested(worker: Any, path: Path, *, label: str) -> dict[str, Any]:
    value = worker.load_json(path, label=label)
    worker.verify_self_digest(value, label=label)
    return value


def _queue_rows(worker: Any, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue_path, _value = worker._verify_binding(manifest.get("queue"), label="queue")
    if worker.sha256_file(queue_path) != QUEUE_FILE_SHA256:
        raise LocalFallbackError("Sealed Page-12 queue bytes drifted.")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(queue_path.read_text().splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 8:
            raise LocalFallbackError(f"Malformed queue row {index}.")
        execution_id, job_path, protocol_path, job_sha, cpus, memory, disk, runtime = fields
        absolute_job = PACKAGE_DIR / worker.safe_relative_path(
            job_path, label=f"queue job {index}"
        )
        if worker.sha256_file(absolute_job) != job_sha:
            raise LocalFallbackError(f"Queue job drifted at row {index}.")
        rows.append(
            {
                "execution_id": execution_id,
                "job_path": job_path,
                "protocol_path": protocol_path,
                "job_file_sha256": job_sha,
                "request_cpus": int(cpus),
                "request_memory_mb": int(memory),
                "request_disk_mb": int(disk),
                "max_runtime_seconds": int(runtime),
            }
        )
    if tuple(row["execution_id"] for row in rows) != EXECUTION_IDS:
        raise LocalFallbackError("Sealed Page-12 queue order or identity drifted.")
    return rows


def _closed_package(worker: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    if worker.sha256_file(manifest_path) != PACKAGE_MANIFEST_FILE_SHA256:
        raise LocalFallbackError("Sealed package-manifest bytes drifted.")
    manifest = _load_digested(worker, manifest_path, label="package manifest")
    source = manifest.get("source_archive")
    if (
        manifest.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("package_id") != PACKAGE_DIR.name
        or manifest.get("status") != "passed_inert_twelve_cells"
        or manifest.get("row_count") != 12
        or tuple(manifest.get("execution_ids", ())) != EXECUTION_IDS
        or manifest.get("execution_target") != "chtc"
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or not isinstance(source, Mapping)
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise LocalFallbackError("Sealed package identity drifted.")
    source_path, _unused = worker._verify_binding(source, label="source archive")
    if worker.sha256_file(source_path) != SOURCE_ARCHIVE_SHA256:
        raise LocalFallbackError("Sealed source archive drifted.")
    controls = manifest.get("control_files")
    if not isinstance(controls, list):
        raise LocalFallbackError("Control-file inventory is absent.")
    for index, row in enumerate(controls):
        worker._verify_binding(row, label=f"control file {index}")
    if worker.sha256_file(PACKAGE_DIR / "run_cell.py") != SEALED_WORKER_SHA256:
        raise LocalFallbackError("Sealed worker bytes drifted.")
    rows = _queue_rows(worker, manifest)
    for row in rows:
        job, package, protocol, _locks = worker._load_closed_job(
            PACKAGE_DIR / row["job_path"]
        )
        execution_id = str(row["execution_id"])
        expected_policy = (
            "append_only" if execution_id.endswith("append_only")
            else "always_commutation_reduced"
        )
        if (
            package.get("sha256") != manifest.get("sha256")
            or job.get("execution_id") != execution_id
            or job.get("comparator_policy") != expected_policy
            or job.get("runtime_insertion_mode")
            != (
                "append_only"
                if expected_policy == "append_only"
                else "full_commutation_reduced"
            )
            or int(job.get("target_horizon", -1)) != TARGET_HORIZON
            or protocol.get("sha256") != job.get("protocol_sha256")
            or int(protocol.get("horizon", -1)) != TARGET_HORIZON
            or protocol.get("request", {}).get("execution", {}).get("resume", {}).get("kind")
            != "fresh_start"
        ):
            raise LocalFallbackError(f"Sealed job/protocol drifted: {execution_id}")
    return manifest, rows


def _physical_memory_bytes() -> int | None:
    try:
        value = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return int(value)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _memory_pressure_percent() -> int | None:
    try:
        value = subprocess.run(
            ["memory_pressure", "-Q"], check=True,
            capture_output=True, text=True,
        ).stdout
        match = re.search(r"free percentage:\s*(\d+)%", value)
        return None if match is None else int(match.group(1))
    except (OSError, subprocess.CalledProcessError):
        return None


def _evidence_memory_bytes(regime: str) -> int:
    row = PRIOR_RESOURCE_EVIDENCE[regime]
    plateau = int(row["plateau_rss_kib"]) * 1024
    global_mib = row["global_singleton_peak_memory_mib"]
    global_bytes = 0 if global_mib is None else int(global_mib) * 1024**2
    return max(plateau, global_bytes)


def _capacity_receipt(worker: Any, output_parent: Path) -> dict[str, Any]:
    physical = _physical_memory_bytes()
    pressure = _memory_pressure_percent()
    existing = output_parent.resolve()
    while not existing.exists():
        existing = existing.parent
    free_disk = shutil.disk_usage(existing).free
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for regime, nph in REGIMES:
            evidence = PRIOR_RESOURCE_EVIDENCE[regime]
            observed_memory = _evidence_memory_bytes(regime)
            required_memory = math.ceil(observed_memory * MEMORY_HEADROOM_FACTOR)
            required_disk = math.ceil(
                int(evidence["disk_usage_kib"]) * 1024 * DISK_HEADROOM_FACTOR
                + 4 * 1024**3
            )
            blockers: list[str] = []
            if physical is None or physical < required_memory:
                blockers.append("physical_memory_below_evidence_guard")
            if pressure is None or pressure < MIN_MEMORY_PRESSURE_FREE_PERCENT:
                blockers.append("memory_pressure_below_guard")
            if free_disk < required_disk:
                blockers.append("free_disk_below_single_row_guard")
            rows.append(
                {
                    "execution_id": _execution_id(regime, nph, policy),
                    "regime_id": regime,
                    "nph": nph,
                    "policy": policy,
                    "prior_evidence": dict(evidence),
                    "observed_peak_memory_bytes": observed_memory,
                    "required_physical_memory_bytes": required_memory,
                    "required_free_disk_bytes": required_disk,
                    "locally_eligible": not blockers,
                    "blockers": blockers,
                }
            )
    retained = 2 * sum(
        int(PRIOR_RESOURCE_EVIDENCE[regime]["disk_usage_kib"]) * 1024
        for regime, _nph in REGIMES
    )
    largest_live = max(
        int(value["disk_usage_kib"]) * 1024
        for value in PRIOR_RESOURCE_EVIDENCE.values()
    )
    all_rows_disk_guard = math.ceil(
        retained * DISK_HEADROOM_FACTOR + largest_live * DISK_HEADROOM_FACTOR
    )
    all_rows_memory_guard = max(
        int(row["required_physical_memory_bytes"]) for row in rows
    )
    all_rows_capable = (
        all(row["locally_eligible"] for row in rows)
        and physical is not None
        and physical >= all_rows_memory_guard
        and free_disk >= all_rows_disk_guard
    )
    return worker.digested(
        {
            "schema": "paper_i_page12_local_fallback_capacity_receipt_v1",
            "status": "passed" if all_rows_capable else "blocked",
            "physical_memory_bytes": physical,
            "memory_pressure_free_percent": pressure,
            "free_disk_bytes": free_disk,
            "required_physical_memory_bytes_for_all_rows": all_rows_memory_guard,
            "required_free_disk_bytes_for_all_rows": all_rows_disk_guard,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "serial_execution_required": True,
            "row_assessments": rows,
            "all_rows_local_capable": all_rows_capable,
            "scheduler_resource_envelopes_are_provenance_not_local_reservations": True,
            "scientific_execution_performed": False,
        }
    )


def _utc(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise LocalFallbackError(f"{label} is not ISO-8601.") from exc
    if parsed.tzinfo is None:
        raise LocalFallbackError(f"{label} lacks a timezone.")
    return parsed.astimezone(timezone.utc)


def _macro_terminal_inventory_is_closed(value: Mapping[str, Any]) -> bool:
    conditional = value.get("conditional_execution_ids")
    terminal = value.get("terminal_chtc_k50_execution_ids")
    eligible = value.get("eligible_k50_continuation_execution_ids")
    stopped = value.get("stop_at_k30_execution_ids")
    closed = value.get("closed_k50_continuation_execution_ids")
    if (
        conditional != list(PAGE16_CONDITIONAL_EXECUTION_IDS)
        or terminal != list(PAGE16_TERMINAL_CHTC_EXECUTION_IDS)
        or not isinstance(eligible, list)
        or not isinstance(stopped, list)
        or not isinstance(closed, list)
        or any(not isinstance(item, str) for item in (*eligible, *stopped))
    ):
        return False
    eligible_set = set(eligible)
    stopped_set = set(stopped)
    expected_set = set(PAGE16_CONDITIONAL_EXECUTION_IDS)
    return (
        len(eligible) == len(eligible_set)
        and len(stopped) == len(stopped_set)
        and not eligible_set.intersection(stopped_set)
        and eligible_set.union(stopped_set) == expected_set
        and eligible
        == [
            execution_id
            for execution_id in PAGE16_CONDITIONAL_EXECUTION_IDS
            if execution_id in eligible_set
        ]
        and stopped
        == [
            execution_id
            for execution_id in PAGE16_CONDITIONAL_EXECUTION_IDS
            if execution_id in stopped_set
        ]
        and closed == eligible
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _macro_terminal_sources_are_pinned(worker: Any) -> bool:
    return (
        MACRO_TERMINAL_PRODUCER_PATH.is_file()
        and not MACRO_TERMINAL_PRODUCER_PATH.is_symlink()
        and worker.sha256_file(MACRO_TERMINAL_PRODUCER_PATH)
        == EXPECTED_MACRO_TERMINAL_PRODUCER_SHA256
        and MACRO_TERMINAL_ADAPTER_PATH.is_file()
        and not MACRO_TERMINAL_ADAPTER_PATH.is_symlink()
        and worker.sha256_file(MACRO_TERMINAL_ADAPTER_PATH)
        == EXPECTED_MACRO_TERMINAL_ADAPTER_SHA256
    )


_MACRO_TERMINAL_REPLAY_SCRIPT = r"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


producer_path = Path(sys.argv[1]).resolve()
adapter_path = Path(sys.argv[2]).resolve()
receipt_path = Path(sys.argv[3]).resolve()
expected_producer_sha256 = sys.argv[4]
expected_adapter_sha256 = sys.argv[5]
for path, expected, label in (
    (producer_path, expected_producer_sha256, "producer"),
    (adapter_path, expected_adapter_sha256, "adapter"),
):
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected
    ):
        raise RuntimeError(f"Pinned macro-terminal {label} bytes drifted.")
if not receipt_path.is_file() or receipt_path.is_symlink():
    raise RuntimeError("Macro-terminal receipt is absent or unsafe.")

spec = importlib.util.spec_from_file_location(
    "paper_i_page12_trusted_macro_terminal_producer", producer_path
)
if spec is None or spec.loader is None:
    raise RuntimeError("Pinned macro-terminal producer cannot be loaded.")
producer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)
if (
    Path(producer.ADAPTER_PATH).resolve() != adapter_path
    or producer.EXPECTED_ADAPTER_SHA256 != expected_adapter_sha256
):
    raise RuntimeError("Producer-to-adapter pin drifted.")
adapter = producer._load_adapter()
if Path(adapter.__file__).resolve() != adapter_path:
    raise RuntimeError("Trusted continuation adapter escaped its pinned path.")
if (
    Path(producer.ACTIVATION_DIR).resolve()
    != Path(adapter.DEFAULT_ACTIVATION_DIR).resolve()
    or Path(producer.RUNTIME_DIR).resolve()
    != Path(adapter.DEFAULT_RUNTIME_DIR).resolve()
):
    raise RuntimeError("Producer continuation activation/runtime paths drifted.")

worker = adapter.k30._load_worker()
activation, bundle = producer._validate_activation(adapter)
runtime, runtime_activation, runtime_bundle = adapter._validate_runtime(
    worker,
    activation_dir=producer.ACTIVATION_DIR,
    runtime_dir=producer.RUNTIME_DIR,
)
if runtime_activation != activation or runtime_bundle != bundle:
    raise RuntimeError("Continuation runtime authority differs from activation.")
k30_runtime = adapter._validated_k30_runtime(worker)
if k30_runtime is None:
    raise RuntimeError("Authenticated k30 runtime is absent.")
snapshot = producer._require_all_decisions(
    adapter.decision_snapshot(cached={})
)
terminal_status = producer._require_all_terminals(
    adapter.terminal_chtc_status(cached={})
)
if (
    runtime.get("activation_manifest_sha256") != activation.get("sha256")
    or runtime.get("k30_runtime_manifest_sha256")
    != k30_runtime.get("sha256")
    or runtime.get("decision_status_sha256") != snapshot.get("sha256")
):
    raise RuntimeError("Live continuation provenance hashes drifted.")
eligible = list(snapshot["eligible_execution_ids"])
unclosed = [
    execution_id
    for execution_id in eligible
    if not adapter.closed_continuation_cell(
        runtime_dir=producer.RUNTIME_DIR,
        execution_id=execution_id,
    )
]
if unclosed:
    raise RuntimeError(
        "Eligible k50 continuation remains unclosed: " + ", ".join(unclosed)
    )

def forbid_receipt_write(*_args, **_kwargs):
    raise RuntimeError("Trusted replay is read-only; receipt write forbidden.")

adapter._write_json = forbid_receipt_write
recomputed = producer._emit_macro_terminal_receipt(
    adapter,
    runtime=runtime,
    activation=activation,
    snapshot=snapshot,
    terminal_status=terminal_status,
    path=receipt_path,
)
expected_hashes = {
    "adapter_sha256": expected_adapter_sha256,
    "activation_manifest_sha256": activation["sha256"],
    "runtime_manifest_sha256": runtime["sha256"],
    "k30_runtime_manifest_sha256": k30_runtime["sha256"],
    "decision_status_sha256": snapshot["sha256"],
    "terminal_chtc_status_sha256": terminal_status["sha256"],
}
if any(recomputed.get(key) != value for key, value in expected_hashes.items()):
    raise RuntimeError("Recomputed macro-terminal dynamic hashes drifted.")
print(
    json.dumps(
        recomputed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ),
    flush=True,
)
"""


def _trusted_macro_terminal_replay(path: Path) -> dict[str, Any]:
    """Recompute the macro closure in a clean, read-only Python process."""

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        }
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                _MACRO_TERMINAL_REPLAY_SCRIPT,
                MACRO_TERMINAL_PRODUCER_PATH.as_posix(),
                MACRO_TERMINAL_ADAPTER_PATH.as_posix(),
                path.as_posix(),
                EXPECTED_MACRO_TERMINAL_PRODUCER_SHA256,
                EXPECTED_MACRO_TERMINAL_ADAPTER_SHA256,
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=MACRO_TERMINAL_REPLAY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise LocalFallbackError(
            "Trusted macro-terminal replay timed out."
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise LocalFallbackError(
            "Trusted macro-terminal replay failed"
            + (f": {detail}" if detail else ".")
        )
    try:
        replay = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LocalFallbackError(
            "Trusted macro-terminal replay returned invalid JSON."
        ) from exc
    if not isinstance(replay, dict):
        raise LocalFallbackError(
            "Trusted macro-terminal replay returned a non-object."
        )
    return replay


def _external_gate(
    worker: Any, path: Path, *, kind: str, now: datetime
) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not path.is_file() or path.is_symlink():
        return False, None, f"{kind}_receipt_absent"
    try:
        value = _load_digested(worker, path, label=f"{kind} receipt")
        if kind == "macro_terminal":
            if (
                not _macro_terminal_sources_are_pinned(worker)
                or value.get("schema") != MACRO_TERMINAL_SCHEMA
                or value.get("status")
                != "passed_all_required_macro_k30_k50_work_terminal"
                or value.get("adapter_sha256")
                != EXPECTED_MACRO_TERMINAL_ADAPTER_SHA256
                or any(
                    not _is_sha256(value.get(field))
                    for field in MACRO_TERMINAL_PROVENANCE_SHA256_FIELDS
                )
                or not _macro_terminal_inventory_is_closed(value)
                or value.get("all_k30_cells_closed") is not True
                or value.get("all_extension_required_cells_closed_at_k50") is not True
                or value.get("remaining_macro_execution_ids") != []
                or value.get("active_macro_execution_ids") != []
                or value.get("scientific_execution_performed_by_receipt") is not False
            ):
                raise LocalFallbackError("Macro terminal receipt drifted.")
            if _trusted_macro_terminal_replay(path) != value:
                raise LocalFallbackError(
                    "Macro terminal receipt differs from trusted replay."
                )
        else:
            observed = _utc(value.get("observed_at_utc"), label="remote observed")
            valid_until = _utc(value.get("valid_until_utc"), label="remote expiry")
            if (
                value.get("schema") != REMOTE_CLEARANCE_SCHEMA
                or value.get("status")
                != "passed_authenticated_no_remote_overlap_clearance"
                or value.get("authentication_kind")
                != "interactive_ssh_duo_condor_q_snapshot_v1"
                or value.get("authenticated_remote_query") is not True
                or value.get("execution_ids") != list(EXECUTION_IDS)
                or value.get("cluster_id") != 9647385
                or value.get("remote_materialized_execution_ids") != []
                or value.get("remote_active_execution_ids") != []
                or value.get("remote_held_execution_ids") != []
                or value.get("remote_latent_execution_ids") != []
                or value.get("factory_present") is not False
                or value.get("overlapping_execution_ids") != []
                or value.get("no_remote_overlap") is not True
                or value.get("scientific_execution_performed") is not False
                or observed > now or now > valid_until
                or valid_until <= observed
                or (valid_until - observed).total_seconds()
                > REMOTE_CLEARANCE_MAX_WINDOW_SECONDS
                or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("scheduler_snapshot_sha256")))
            ):
                raise LocalFallbackError("Remote-overlap receipt drifted.")
        return True, value, None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, LocalFallbackError) as exc:
        return False, None, f"{kind}_receipt_invalid:{exc}"


def _activation_binding(worker: Any, root: Path, path: Path) -> dict[str, Any]:
    value = _load_digested(worker, path, label=path.name)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": worker.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "canonical_sha256": value["sha256"],
    }


def prepare_activation(*, activation_dir: Path, output_parent: Path) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_package(worker)
    if activation_dir.exists() or activation_dir.is_symlink():
        raise FileExistsError(f"Activation destination exists: {activation_dir}")
    activation_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{activation_dir.name}.build-", dir=activation_dir.parent))
    try:
        request = worker.digested(
            {
                "schema": LOCAL_REQUEST_SCHEMA,
                "status": "prepared_pending_external_gates",
                "source_package_id": manifest["package_id"],
                "source_campaign_id": manifest["campaign_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": worker.sha256_file(ADAPTER_PATH),
                "execution_ids": list(EXECUTION_IDS),
                "target_horizon": TARGET_HORIZON,
                "maximum_concurrency": MAXIMUM_CONCURRENCY,
                "serial_execution_required": True,
                "macro_k30_k50_terminal_gate_required": True,
                "fresh_no_remote_overlap_gate_required": True,
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
                "scientific_execution_performed": False,
            }
        )
        request_path = temporary / "activation_request.json"
        _write_json_exclusive(worker, request_path, request)
        authorizations: list[dict[str, Any]] = []
        for row in rows:
            job, _package, protocol, _locks = worker._load_closed_job(
                PACKAGE_DIR / row["job_path"]
            )
            authority = worker.digested(
                {
                    "schema": LOCAL_AUTHORIZATION_SCHEMA,
                    "status": "pending_external_terminal_overlap_and_capacity_gates",
                    "execution_id": job["execution_id"],
                    "regime_id": job["regime_id"],
                    "nph": int(job["nph"]),
                    "comparator_policy": job["comparator_policy"],
                    "runtime_insertion_mode": job["runtime_insertion_mode"],
                    "job_spec_sha256": job["sha256"],
                    "job_file_sha256": row["job_file_sha256"],
                    "protocol_sha256": protocol["sha256"],
                    "package_manifest_sha256": manifest["sha256"],
                    "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                    "target_horizon": TARGET_HORIZON,
                    "fresh_start": True,
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "scientific_execution_performed": False,
                }
            )
            path = temporary / "authorizations" / f"{job['execution_id']}.json"
            _write_json_exclusive(worker, path, authority)
            authorizations.append(
                {"execution_id": job["execution_id"], **_activation_binding(worker, temporary, path)}
            )
        capacity = _capacity_receipt(worker, output_parent)
        capacity_path = temporary / "capacity_at_preparation.json"
        _write_json_exclusive(worker, capacity_path, capacity)
        activation = worker.digested(
            {
                "schema": LOCAL_ACTIVATION_SCHEMA,
                "status": "passed_dormant_local_fallback_materialization",
                "source_package_id": manifest["package_id"],
                "package_manifest_sha256": manifest["sha256"],
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
                "local_adapter_sha256": worker.sha256_file(ADAPTER_PATH),
                "activation_request": _activation_binding(worker, temporary, request_path),
                "capacity_at_preparation": _activation_binding(worker, temporary, capacity_path),
                "authorizations": authorizations,
                "authorization_count": 12,
                "execution_ids": list(EXECUTION_IDS),
                "target_horizon": TARGET_HORIZON,
                "maximum_concurrency": MAXIMUM_CONCURRENCY,
                "serial_execution_required": True,
                "execution_entrypoint_present": False,
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        _write_json_exclusive(worker, temporary / "activation_manifest.json", activation)
        os.rename(temporary, activation_dir)
        return activation
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_activation(worker: Any, activation_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    activation = _load_digested(worker, activation_dir / "activation_manifest.json", label="local activation")
    if (
        activation_dir.is_symlink()
        or activation.get("schema") != LOCAL_ACTIVATION_SCHEMA
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or activation.get("local_adapter_sha256") != worker.sha256_file(ADAPTER_PATH)
        or activation.get("execution_ids") != list(EXECUTION_IDS)
        or activation.get("authorization_count") != 12
        or activation.get("execution_entrypoint_present") is not False
        or activation.get("execution_authorized") is not False
        or activation.get("submission_authorized") is not False
    ):
        raise LocalFallbackError("Dormant activation drifted.")
    return activation


def inert_preflight(
    *, activation_dir: Path, output_parent: Path,
    macro_terminal_receipt: Path, remote_clearance: Path,
) -> dict[str, Any]:
    worker = _load_worker()
    manifest, rows = _closed_package(worker)
    sealed = []
    for row in rows:
        receipt = worker.preflight(PACKAGE_DIR / row["job_path"])
        if (
            receipt.get("status") != "passed"
            or receipt.get("execution_id") != row["execution_id"]
            or receipt.get("target_horizon") != TARGET_HORIZON
            or receipt.get("fresh_start") is not True
            or receipt.get("scientific_execution_performed") is not False
        ):
            raise LocalFallbackError(f"Worker preflight drifted: {row['execution_id']}")
        sealed.append(receipt)
    capacity = _capacity_receipt(worker, output_parent)
    now = datetime.now(timezone.utc)
    macro_ok, macro_value, macro_blocker = _external_gate(
        worker, macro_terminal_receipt, kind="macro_terminal", now=now
    )
    remote_ok, remote_value, remote_blocker = _external_gate(
        worker, remote_clearance, kind="remote_overlap", now=now
    )
    activation_status = "absent"
    if activation_dir.exists() or activation_dir.is_symlink():
        _validate_activation(worker, activation_dir, manifest)
        activation_status = "validated_dormant"
    blockers = [
        blocker for blocker in (
            None if capacity["status"] == "passed" else "host_capacity_blocked",
            macro_blocker,
            remote_blocker,
            None if activation_status == "validated_dormant" else "activation_absent",
            "execution_entrypoint_intentionally_absent",
        ) if blocker is not None
    ]
    return worker.digested(
        {
            "schema": LOCAL_PREFLIGHT_SCHEMA,
            "status": "passed_inert_preflight_with_blockers" if blockers else "passed_inert_preflight",
            "source_package_id": manifest["package_id"],
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "local_adapter_sha256": worker.sha256_file(ADAPTER_PATH),
            "execution_ids": list(EXECUTION_IDS),
            "sealed_worker_preflight_count": len(sealed),
            "sealed_worker_preflights": sealed,
            "capacity": capacity,
            "macro_terminal_gate": {
                "path": macro_terminal_receipt.as_posix(),
                "passed": macro_ok,
                "receipt_sha256": None if macro_value is None else macro_value["sha256"],
                "requires_all_k30_and_every_required_k50_continuation": True,
            },
            "remote_overlap_gate": {
                "path": remote_clearance.as_posix(),
                "passed": remote_ok,
                "receipt_sha256": None if remote_value is None else remote_value["sha256"],
                "requires_cluster_9647385_and_factory_absent": True,
            },
            "activation_status": activation_status,
            "maximum_concurrency": MAXIMUM_CONCURRENCY,
            "serial_execution_required": True,
            "fallback_decision": (
                "prefer_chtc_current_host_infeasible"
                if capacity["status"] != "passed"
                else "local_fallback_may_be_materialized_after_external_gates"
            ),
            "run_ready": False,
            "blockers": blockers,
            "execution_entrypoint_present": False,
            "scientific_execution_performed": False,
            "submission_performed": False,
            "scheduler_state_changed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dormant local-only Page-12 insertion-comparator fallback"
    )
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument(
        "--output-parent", type=Path,
        default=REPO_ROOT / "output/local_runs/paper_i_page12_insertion_comparators_r50_20260812_v1",
    )
    parser.add_argument(
        "--macro-terminal-receipt", type=Path,
        default=DEFAULT_MACRO_TERMINAL_RECEIPT,
    )
    parser.add_argument("--remote-clearance", type=Path, default=DEFAULT_REMOTE_CLEARANCE)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            payload = prepare_activation(
                activation_dir=args.activation_dir.resolve(),
                output_parent=args.output_parent.resolve(),
            )
        else:
            payload = inert_preflight(
                activation_dir=args.activation_dir.resolve(),
                output_parent=args.output_parent.resolve(),
                macro_terminal_receipt=args.macro_terminal_receipt.resolve(),
                remote_clearance=args.remote_clearance.resolve(),
            )
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError, LocalFallbackError
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_load_worker().canonical_json_bytes(payload).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
