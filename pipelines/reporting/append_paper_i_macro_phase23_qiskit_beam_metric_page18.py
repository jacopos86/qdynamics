#!/usr/bin/env python3
"""Build the schema-locked Page-18 macro Qiskit beam comparison.

This updater is deliberately local and reporting-only.  It never contacts
CHTC or changes scheduler state.  Completed cluster-9649696 cells are accepted
only through a self-digested retrieval manifest and a complete archive closure
against the sealed package, job, worker receipt, execution manifest, and every
declared artifact.  A run that stops before or after controller round 20 is
rejected.

The default command writes a standalone Page-18 PDF/PNG/adapter/provenance.
It does not mutate the canonical report.  ``--append-canonical`` is a separate
fail-closed operation: all six cells must be complete, the Page-17 watcher must
be absent, and all existing Pages 1--17 must remain content-stream identical.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import uuid
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    append_paper_i_completed_beam_noise_pages as completed_pages,
)
from pipelines.reporting import (  # noqa: E402
    append_paper_i_macro_phase0_proxy_no_lanes_page13 as page13,
)
from pipelines.reporting import (  # noqa: E402
    append_paper_i_macro_phase23_qiskit_no_lanes_page16 as page16,
)


REPORT_DIR = REPO_ROOT / (
    "output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving"
)
TARGET_PDF = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress.pdf"
)
TARGET_PROVENANCE = TARGET_PDF.with_name(
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_provenance.json"
)
PAGE_STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "macro_phase23_qiskit_beam3x2_metric_page18"
)
PAGE_PDF = REPORT_DIR / f"{PAGE_STEM}.pdf"
PAGE_PNG = REPORT_DIR / f"{PAGE_STEM}.png"
ADAPTER_PATH = REPORT_DIR / f"{PAGE_STEM}_adapter.json"
STANDALONE_PROVENANCE = REPORT_DIR / f"{PAGE_STEM}_provenance.json"
PAGE_ID = "macro_phase23_qiskit_beam3x2_metric_prune_k20_comparison_v1"

PACKAGE_ID = (
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_20260812_"
    "v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_v2"
)
BUNDLE_ID = (
    "ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_v2"
)
PACKAGE_DIR = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727" / PACKAGE_ID
SUBMISSION_RECEIPT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_r20_20260812_"
    "v2_submission_receipt_9649696.json"
)
RETRIEVED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260813_page16_macro_qiskit_beam_metric_v2"
)
RETRIEVAL_MANIFEST = RETRIEVED_DIR / "retrieval_manifest.json"

PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "24d3b7f73554ec274acbb6f2649082d5d0d52ec85ab68a60e2a1584c46a6559d"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "114477e3c415702d036c055216265ab4580d57d40fb859b73fc04603bc0ae5a1"
)
SOURCE_ARCHIVE_SHA256 = (
    "47baf7280cf763df57c3d23600500082f6054ccef887208bf25c928630544c8d"
)
SOURCE_ARCHIVE_MANIFEST_SHA256 = (
    "61ba8366f568495f2fd0e72362f5b62fa9f1316716cc55c1c634fe246f286fbc"
)
ROUTE_CONTRACT_SHA256 = (
    "62dd2b102d7b664121c9265e1b7e2e97382d2acb8fdcfe7238ad9ae28720d452"
)
SUBMISSION_RECEIPT_SHA256 = (
    "745e1a46e285dc9672972e897b1207d022c2dd62781cc1b57715e19ab2a9e702"
)
CLUSTER_ID = 9649696
TARGET_HORIZON = 20

PAGE14_ADAPTER = completed_pages.PAGE14_ADAPTER
PAGE16_ADAPTER = page16.ADAPTER_PATH
PAGE14_ADAPTER_CANONICAL_SHA256 = (
    "4278cb6cf2e7abb052b97c77f65d45cb3c2c873f8a7a4e8570789700dab0f214"
)
PAGE16_ADAPTER_CANONICAL_SHA256 = (
    "a958845b35d71737adeb5d9dceb6cbbf52ea2b35de581a32cf21d3fefd26b139"
)

RETRIEVAL_SCHEMA = "paper_i_page18_macro_qiskit_beam_verified_retrieval_v1"
ADAPTER_SCHEMA = "paper_i_macro_phase23_qiskit_beam_metric_page18_adapter_v1"
STANDALONE_SCHEMA = (
    "paper_i_macro_phase23_qiskit_beam_metric_page18_standalone_report_v1"
)
MASTER_REPORT_SCHEMA = (
    "paper_i_macro_phase23_qiskit_beam_metric_page18_master_append_v1"
)

REGIME_ORDER = page13.REGIME_ORDER
REGIME_LABELS = page13.REGIME_LABELS
NPH = page13.NPH

WATCHER_NAME = "watch_paper_i_page16_insertion_comparator_snapshot.py"
WATCH_LOCK = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "insertion_comparator_live_snapshot_watch.lock"
)
PAGE16_ID = page16.PAGE_ID
PAGE17_ID = "phase0_insertion_comparator_page16_six_regime_progress_snapshot_v3"

PLOT_FLOOR = 1.0e-16
BLUE = "#4C78A8"
PURPLE = "#CC79A7"
GREEN = "#009E73"
ORANGE = "#E69F00"


class UpdateError(ValueError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UpdateError(f"cannot load JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise UpdateError(f"JSON object required: {path}")
    return value


def binding(path: Path) -> dict[str, Any]:
    return completed_pages.binding(path)


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if claimed != _canonical_sha256(unsigned):
        raise UpdateError(f"{label}: self digest drifted")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_member_name(name: str) -> str:
    raw = str(name)
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise UpdateError(f"unsafe archive member: {name}")
    return path.as_posix()


def _regular_bound_file(root: Path, row: Mapping[str, Any], *, label: str) -> Path:
    relative = _safe_member_name(str(row.get("path", "")))
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise UpdateError(f"{label}: bound file is missing or unsafe")
    actual = binding(path)
    if (
        actual["sha256"] != row.get("sha256")
        or actual["size_bytes"] != row.get("size_bytes")
    ):
        raise UpdateError(f"{label}: file binding drifted")
    return path


def _digested_bound_file(
    root: Path, row: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = _regular_bound_file(root, row, label=label)
    value = load(path)
    verify_self_digest(value, label=label)
    if value.get("sha256") != row.get("canonical_sha256"):
        raise UpdateError(f"{label}: canonical binding drifted")
    return value


def _validate_source_archive(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source_row = manifest.get("source_archive")
    source_manifest_row = manifest.get("source_archive_manifest")
    if not isinstance(source_row, Mapping) or not isinstance(
        source_manifest_row, Mapping
    ):
        raise UpdateError("sealed source bindings are absent")
    archive_path = _regular_bound_file(
        PACKAGE_DIR, source_row, label="source archive"
    )
    if source_row.get("sha256") != SOURCE_ARCHIVE_SHA256:
        raise UpdateError("source archive identity drifted")
    source_manifest = _digested_bound_file(
        PACKAGE_DIR,
        source_manifest_row,
        label="source archive manifest",
    )
    rows = source_manifest.get("members")
    archive_binding = source_manifest.get("archive")
    if (
        source_manifest.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_source_archive_manifest_v1"
        or source_manifest.get("status") != "passed"
        or source_manifest.get("package_id") != PACKAGE_ID
        or source_manifest.get("sha256") != SOURCE_ARCHIVE_MANIFEST_SHA256
        or not isinstance(archive_binding, Mapping)
        or archive_binding.get("path") != source_row.get("path")
        or archive_binding.get("sha256") != SOURCE_ARCHIVE_SHA256
        or archive_binding.get("size_bytes") != source_row.get("size_bytes")
        or not isinstance(rows, list)
        or int(source_manifest.get("member_count", -1)) != len(rows)
    ):
        raise UpdateError("source archive manifest identity drifted")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise UpdateError("invalid source archive member row")
        relative = _safe_member_name(str(row.get("path", "")))
        if relative in declared:
            raise UpdateError(f"duplicate source archive member: {relative}")
        declared[relative] = row
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = _safe_member_name(member.name)
            if (
                relative in observed
                or relative not in declared
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise UpdateError(f"unsafe/undeclared source member: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise UpdateError(f"unreadable source member: {relative}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
            row = declared[relative]
            if digest.hexdigest() != row.get("sha256") or size != row.get(
                "size_bytes"
            ):
                raise UpdateError(f"source member binding drifted: {relative}")
            observed.add(relative)
    if observed != set(declared):
        raise UpdateError("source archive membership is incomplete")
    return source_manifest


def _validate_package_authority() -> dict[str, Any]:
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    if binding(manifest_path)["sha256"] != PACKAGE_MANIFEST_FILE_SHA256:
        raise UpdateError("package manifest file bytes drifted")
    manifest = load(manifest_path)
    verify_self_digest(manifest, label="Page-18 source package manifest")
    if (
        manifest.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_package_manifest_v1"
        or manifest.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("status") != "passed_inert_six_cells"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("child_route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("submitted") is not False
    ):
        raise UpdateError("Page-18 sealed package identity drifted")
    _validate_source_archive(manifest)
    for key, label in (
        ("execution_plan", "execution plan"),
        ("bundle_manifest", "bundle manifest"),
        ("bundle_source_locks", "bundle source locks"),
        ("bundle_validation_report", "bundle validation report"),
        ("bundle_expected_artifacts", "expected artifacts"),
    ):
        row = manifest.get(key)
        if not isinstance(row, Mapping):
            raise UpdateError(f"{label} binding is absent")
        _digested_bound_file(PACKAGE_DIR, row, label=label)
    queue_row = manifest.get("queue")
    if not isinstance(queue_row, Mapping):
        raise UpdateError("queue binding is absent")
    _regular_bound_file(PACKAGE_DIR, queue_row, label="queue")
    receipt_binding = binding(SUBMISSION_RECEIPT)
    if receipt_binding["sha256"] != SUBMISSION_RECEIPT_SHA256:
        raise UpdateError("cluster-9649696 submission receipt bytes drifted")
    receipt = load(SUBMISSION_RECEIPT)
    if (
        receipt.get("cluster_id") != CLUSTER_ID
        or receipt.get("execution_count") != 6
        or receipt.get("execution_scope") != "l2_macro_only_six_regimes_k20"
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("package_manifest_canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or receipt.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or receipt.get("remote_deep_worker_preflight_status") != "passed"
        or receipt.get("remote_deep_worker_preflight_count") != 6
    ):
        raise UpdateError("cluster-9649696 submission receipt identity drifted")
    return manifest


def _package_jobs(
    manifest: Mapping[str, Any],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    rows = manifest.get("jobs")
    protocol_rows = manifest.get("protocols")
    expected_row = manifest.get("bundle_expected_artifacts")
    if (
        not isinstance(rows, list)
        or len(rows) != 6
        or not isinstance(protocol_rows, list)
        or len(protocol_rows) != 6
        or not isinstance(expected_row, Mapping)
    ):
        raise UpdateError("package job inventory is not exactly six cells")
    protocol_by_id = {
        str(row.get("execution_id")): row
        for row in protocol_rows
        if isinstance(row, Mapping)
    }
    if len(protocol_by_id) != 6:
        raise UpdateError("package protocol inventory is not exactly six cells")
    expected_artifacts = _digested_bound_file(
        PACKAGE_DIR, expected_row, label="expected artifacts"
    )
    expected_cells = expected_artifacts.get("cells")
    if (
        expected_artifacts.get("schema")
        != "page16_macro_phase23_qiskit_beam_metric_expected_artifacts_v1"
        or expected_artifacts.get("bundle_id") != BUNDLE_ID
        or not isinstance(expected_cells, Mapping)
        or len(expected_cells) != 6
    ):
        raise UpdateError("expected-artifact authority drifted")
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise UpdateError("package job binding is invalid")
        path = _regular_bound_file(PACKAGE_DIR, row, label="job")
        job = load(path)
        verify_self_digest(job, label=f"job {path.name}")
        execution_id = str(job.get("execution_id", ""))
        protocol_row = protocol_by_id.get(execution_id)
        expected_cell = expected_cells.get(execution_id)
        if not isinstance(protocol_row, Mapping) or not isinstance(
            expected_cell, Mapping
        ):
            raise UpdateError(f"job authorities are absent: {execution_id}")
        protocol_path = _regular_bound_file(
            PACKAGE_DIR, protocol_row, label=f"protocol {execution_id}"
        )
        protocol = load(protocol_path)
        verify_self_digest(protocol, label=f"protocol {execution_id}")
        route_contract = protocol.get("route_contract")
        if not isinstance(route_contract, Mapping):
            raise UpdateError(f"protocol route contract is absent: {execution_id}")
        verify_self_digest(route_contract, label=f"route contract {execution_id}")
        if (
            job.get("schema")
            != "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_job_v1"
            or job.get("package_id") != PACKAGE_ID
            or job.get("campaign_id") != CAMPAIGN_ID
            or job.get("bundle_id") != BUNDLE_ID
            or job.get("target_horizon") != TARGET_HORIZON
            or job.get("candidate_representation") != "macro_generator_v1"
            or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
            or job.get("route_id")
            != "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_plateau_beam3x2_metric_prune"
            or job.get("selector_compile_cost_scope")
            != "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
            or protocol.get("schema") != "paper_i_ra_adapt_resolved_protocol_v1"
            or protocol.get("sha256") != protocol_row.get("canonical_sha256")
            or protocol.get("sha256") != job.get("protocol_sha256")
            or protocol_row.get("path") != job.get("protocol_path")
            or protocol_row.get("sha256") != job.get("protocol_file_sha256")
            or protocol.get("bundle_id") != BUNDLE_ID
            or protocol.get("horizon") != TARGET_HORIZON
            or protocol.get("candidate_representation") != "macro_generator_v1"
            or protocol.get("optimizer") != "powell"
            or protocol.get("optimizer_maxiter") != 200
            or protocol.get("seeds") != {"adapt": 7, "transpiler": 7}
            or route_contract.get("sha256") != ROUTE_CONTRACT_SHA256
            or expected_cell.get("expected_run_artifacts")
            != job.get("expected_run_artifacts")
            or execution_id in result
        ):
            raise UpdateError(f"job scientific identity drifted: {path}")
        result[execution_id] = (path, job)
    if set(result) != set(manifest.get("execution_ids", [])):
        raise UpdateError("package execution-id closure drifted")
    return result


def _validate_retrieval_manifest(
    jobs: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    if not RETRIEVAL_MANIFEST.exists() and not RETRIEVAL_MANIFEST.is_symlink():
        return {}
    if not RETRIEVAL_MANIFEST.is_file() or RETRIEVAL_MANIFEST.is_symlink():
        raise UpdateError("retrieval manifest is missing or unsafe")
    manifest = load(RETRIEVAL_MANIFEST)
    verify_self_digest(manifest, label="Page-18 retrieval manifest")
    rows = manifest.get("archives")
    if (
        manifest.get("schema") != RETRIEVAL_SCHEMA
        or manifest.get("cluster_id") != CLUSTER_ID
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("package_manifest_canonical_sha256")
        != PACKAGE_MANIFEST_CANONICAL_SHA256
        or manifest.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or not isinstance(rows, list)
        or manifest.get("status")
        not in {"partial_verified_fetches", "verified_six_archives"}
    ):
        raise UpdateError("Page-18 retrieval manifest identity drifted")
    by_id: dict[str, Mapping[str, Any]] = {}
    proc_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise UpdateError("retrieval archive row is invalid")
        execution_id = str(row.get("execution_id", ""))
        proc_id = int(row.get("proc_id", -1))
        if (
            execution_id not in jobs
            or execution_id in by_id
            or proc_id in proc_ids
            or not 0 <= proc_id < 6
            or row.get("fetch_verification")
            != "remote_and_local_size_sha256_match_v1"
        ):
            raise UpdateError("retrieval archive identity is invalid or duplicated")
        expected_filename = f"{execution_id}__{CLUSTER_ID}__{proc_id}.tar.gz"
        if (
            row.get("filename") != expected_filename
            or not str(row.get("remote_path", "")).endswith(
                f"/transfer/{expected_filename}"
            )
            or not isinstance(row.get("sha256"), str)
            or len(str(row.get("sha256"))) != 64
            or int(row.get("size_bytes", -1)) <= 0
        ):
            raise UpdateError(f"retrieval archive binding drifted: {execution_id}")
        path = RETRIEVED_DIR / expected_filename
        actual = binding(path)
        if (
            actual["sha256"] != row["sha256"]
            or actual["size_bytes"] != row["size_bytes"]
        ):
            raise UpdateError(f"retrieved archive bytes drifted: {execution_id}")
        by_id[execution_id] = row
        proc_ids.add(proc_id)
    if len(rows) != int(manifest.get("archive_count", -1)):
        raise UpdateError("retrieval archive count drifted")
    expected_status = (
        "verified_six_archives" if len(rows) == 6 else "partial_verified_fetches"
    )
    if manifest.get("status") != expected_status:
        raise UpdateError("retrieval manifest completion status drifted")
    return by_id


def _sha256_stream(stream: Any) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured: bytes | None = b""
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if captured is not None:
            if size <= 4 * 1024 * 1024:
                captured += block
            else:
                captured = None
    return digest.hexdigest(), size, captured


def _json_capture(
    observed: Mapping[str, Mapping[str, Any]], relative: str, *, label: str
) -> dict[str, Any]:
    raw = observed.get(relative, {}).get("captured")
    if not isinstance(raw, bytes):
        raise UpdateError(f"{label} is absent or unexpectedly large")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UpdateError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise UpdateError(f"{label} must be a JSON object")
    return value


def _close_archive(
    *,
    path: Path,
    retrieval: Mapping[str, Any],
    job_path: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id = str(job["execution_id"])
    observed: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative = _safe_member_name(member.name)
            if relative in observed or relative in directories:
                raise UpdateError(f"duplicate archive member: {relative}")
            if member.issym() or member.islnk():
                raise UpdateError(f"linked archive member is forbidden: {relative}")
            if member.isdir():
                directories.add(relative)
                continue
            if not member.isfile():
                raise UpdateError(f"unsafe archive member type: {relative}")
            stream = archive.extractfile(member)
            if stream is None:
                raise UpdateError(f"unreadable archive member: {relative}")
            digest, size, captured = _sha256_stream(stream)
            observed[relative] = {
                "sha256": digest,
                "size_bytes": size,
                "captured": captured,
            }

    roots = {"worker_exit_status.txt", "worker_receipt.json"}
    if not roots.issubset(observed):
        raise UpdateError("worker root receipts are absent")
    exit_raw = observed["worker_exit_status.txt"]["captured"]
    if not isinstance(exit_raw, bytes) or exit_raw.strip() != b"0":
        raise UpdateError("worker exit status is nonzero or unreadable")
    worker = _json_capture(observed, "worker_receipt.json", label="worker receipt")
    verify_self_digest(worker, label="worker receipt")
    if (
        worker.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_worker_receipt_v1"
        or worker.get("status") != "passed"
        or worker.get("package_id") != PACKAGE_ID
        or worker.get("campaign_id") != CAMPAIGN_ID
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job["sha256"]
        or worker.get("controller_rounds_completed") != TARGET_HORIZON
        or worker.get("fresh_start") is not True
    ):
        raise UpdateError(f"worker identity/horizon drifted: {execution_id}")
    raw_artifacts = worker.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise UpdateError("worker artifact inventory is absent")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in raw_artifacts:
        if not isinstance(row, Mapping):
            raise UpdateError("worker artifact row is invalid")
        relative = _safe_member_name(str(row.get("path", "")))
        if relative in declared:
            raise UpdateError(f"duplicate worker artifact: {relative}")
        declared[relative] = row
    expected_roles = job.get("expected_run_artifacts")
    if not isinstance(expected_roles, Mapping) or set(expected_roles) != {
        "checkpoint",
        "estimator_ledger",
        "execution_manifest",
        "result",
        "summary",
    }:
        raise UpdateError("job expected-artifact roles drifted")
    expected_paths = {
        _safe_member_name(str(row["path"]))
        for row in expected_roles.values()
        if isinstance(row, Mapping)
    }
    if set(declared) != expected_paths or set(observed) != roots | expected_paths:
        raise UpdateError("archive artifact membership is not exactly closed")
    for relative, row in declared.items():
        actual = observed[relative]
        if (
            actual["sha256"] != row.get("sha256")
            or actual["size_bytes"] != row.get("size_bytes")
        ):
            raise UpdateError(f"artifact binding drifted: {relative}")

    manifest_name = str(expected_roles["execution_manifest"]["path"])
    summary_name = str(expected_roles["summary"]["path"])
    execution_manifest = _json_capture(
        observed, manifest_name, label="execution manifest"
    )
    verify_self_digest(execution_manifest, label="execution manifest")
    if (
        execution_manifest.get("schema")
        != "paper_i_ra_adapt_page16_macro_phase23_qiskit_beam_metric_execution_manifest_v1"
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("package_id") != PACKAGE_ID
        or execution_manifest.get("campaign_id") != CAMPAIGN_ID
        or execution_manifest.get("execution_id") != execution_id
        or execution_manifest.get("job_spec_sha256") != job["sha256"]
        or execution_manifest.get("route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or execution_manifest.get("target_horizon") != TARGET_HORIZON
        or execution_manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or execution_manifest.get("fresh_start") is not True
        or execution_manifest.get("source_checkpoint_consumed") is not False
        or execution_manifest.get("sha256")
        != worker.get("execution_manifest_sha256")
        or execution_manifest.get("authorization_sha256")
        != worker.get("authorization_sha256")
    ):
        raise UpdateError(f"execution closure drifted: {execution_id}")
    payloads = execution_manifest.get("output_payloads")
    if not isinstance(payloads, Mapping) or set(payloads) != {
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    }:
        raise UpdateError("execution output-payload roles drifted")
    for role, row in payloads.items():
        if not isinstance(row, Mapping):
            raise UpdateError(f"execution payload row is invalid: {role}")
        relative = _safe_member_name(str(row.get("path", "")))
        actual = observed.get(relative)
        if (
            actual is None
            or relative != expected_roles[role]["path"]
            or actual["sha256"] != row.get("sha256")
            or actual["size_bytes"] != row.get("size_bytes")
        ):
            raise UpdateError(f"execution payload binding drifted: {role}")

    summary = _json_capture(observed, summary_name, label="Paper-I summary")
    trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or not isinstance(trace, list)
        or [row.get("controller_round") for row in trace]
        != list(range(1, TARGET_HORIZON + 1))
    ):
        raise UpdateError(f"accepted trajectory is not exactly k=20: {execution_id}")
    exact = float(summary["provenance"]["exact_same_cutoff_energy"])
    points: list[dict[str, Any]] = []
    for row in trace:
        if not math.isclose(
            float(row["exact_same_cutoff_energy"]),
            exact,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise UpdateError(f"same-cutoff reference drifted: {execution_id}")
        points.append(
            {
                "k": int(row["controller_round"]),
                "energy": float(row["accepted_energy"]),
                "error": float(row["absolute_energy_error"]),
                "active_ansatz_depth": int(row["active_ansatz_depth"]),
            }
        )
    costs, compile_receipt = completed_pages._compile_cost_tuple(
        summary, round_index=TARGET_HORIZON
    )
    return {
        "status": "completed_authenticated_chtc_archive",
        "cluster_id": CLUSTER_ID,
        "proc_id": int(retrieval["proc_id"]),
        "execution_id": execution_id,
        "target_horizon": TARGET_HORIZON,
        "points": points,
        "terminal": copy.deepcopy(points[-1]),
        "costs": costs,
        "compile": compile_receipt,
        "sources": {
            "archive": binding(path),
            "retrieval": copy.deepcopy(dict(retrieval)),
            "job": binding(job_path),
            "worker_receipt": {
                "canonical_sha256": worker["sha256"],
                "file_sha256": observed["worker_receipt.json"]["sha256"],
            },
            "execution_manifest": {
                "canonical_sha256": execution_manifest["sha256"],
                "file_sha256": observed[manifest_name]["sha256"],
            },
            "summary": {
                "member": summary_name,
                "file_sha256": observed[summary_name]["sha256"],
            },
            "closure": {
                "worker_exit_status": 0,
                "declared_artifact_count": len(declared),
                "all_declared_artifact_hashes_verified": True,
                "unbound_file_count": 0,
                "exact_controller_rounds": TARGET_HORIZON,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            },
        },
    }


def _validated_comparators() -> tuple[dict[str, Any], dict[str, Any]]:
    page14_adapter = load(PAGE14_ADAPTER)
    page16_adapter = load(PAGE16_ADAPTER)
    for value, label, expected_sha in (
        (page14_adapter, "Page-14 adapter", PAGE14_ADAPTER_CANONICAL_SHA256),
        (page16_adapter, "Page-16 adapter", PAGE16_ADAPTER_CANONICAL_SHA256),
    ):
        verify_self_digest(value, label=label)
        if value.get("sha256") != expected_sha:
            raise UpdateError(f"{label} identity drifted")
    if (
        page14_adapter.get("schema")
        != "paper_i_macro_phase0_beam_metric_page14_adapter_v1"
        or page14_adapter.get("page_id") != completed_pages.PAGE14_ID
        or page14_adapter.get("status") != "completed_6_of_6"
        or page16_adapter.get("schema")
        != "paper_i_macro_phase0_phase23_qiskit_no_lanes_page16_adapter_v1"
        or page16_adapter.get("page_id") != PAGE16_ID
        or page16_adapter.get("status") != "completed_6_of_6_mixed_horizon"
    ):
        raise UpdateError("Page-14/Page-16 comparison authority drifted")
    for adapter, route_key, label in (
        (page14_adapter, "beam_metric_route", "Page-14 proxy beam"),
        (page16_adapter, "page16_qiskit_route", "Page-16 Qiskit"),
    ):
        cells = adapter.get("cells")
        if (
            not isinstance(cells, list)
            or [row.get("regime_id") for row in cells] != list(REGIME_ORDER)
        ):
            raise UpdateError(f"{label} regime coverage drifted")
        for cell in cells:
            route = cell.get(route_key)
            adapt = cell.get("conventional_unwhitened_adapt")
            if not isinstance(route, Mapping) or not isinstance(adapt, Mapping):
                raise UpdateError(f"{label} completed route is absent")
            _prefix_points(route, label=label)
            _prefix_points(adapt, label="conventional ADAPT")
    return page14_adapter, page16_adapter


def _prefix_points(source: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    raw = source.get("points")
    if not isinstance(raw, list):
        raise UpdateError(f"{label}: point sequence is absent")
    points = [copy.deepcopy(dict(row)) for row in raw if int(row.get("k", -1)) <= 20]
    if [int(row.get("k", -1)) for row in points] != list(range(1, 21)):
        raise UpdateError(f"{label}: exact k=1..20 trajectory is unavailable")
    return points


def build_adapter() -> dict[str, Any]:
    manifest = _validate_package_authority()
    jobs = _package_jobs(manifest)
    retrievals = _validate_retrieval_manifest(jobs)
    page14_adapter, page16_adapter = _validated_comparators()
    page14_cells = {row["regime_id"]: row for row in page14_adapter["cells"]}
    page16_cells = {row["regime_id"]: row for row in page16_adapter["cells"]}
    cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        matches = [
            value
            for execution_id, value in jobs.items()
            if f"__{regime}__nph{NPH[regime]}__" in execution_id
        ]
        if len(matches) != 1:
            raise UpdateError(f"Page-18 job coverage drifted: {regime}")
        job_path, job = matches[0]
        if job.get("regime_id") != regime or job.get("nph") != NPH[regime]:
            raise UpdateError(f"Page-18 job regime drifted: {regime}")
        route = None
        retrieval = retrievals.get(str(job["execution_id"]))
        if retrieval is not None:
            route = _close_archive(
                path=RETRIEVED_DIR / str(retrieval["filename"]),
                retrieval=retrieval,
                job_path=job_path,
                job=job,
            )
        p14 = page14_cells[regime]
        p16 = page16_cells[regime]
        adapt14 = _prefix_points(
            p14["conventional_unwhitened_adapt"], label="Page-14 ADAPT"
        )
        adapt16 = _prefix_points(
            p16["conventional_unwhitened_adapt"], label="Page-16 ADAPT"
        )
        if adapt14 != adapt16:
            raise UpdateError(f"conventional ADAPT comparator drifted: {regime}")
        cells.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": NPH[regime],
                "target_horizon": TARGET_HORIZON,
                "conventional_unwhitened_adapt": {
                    **copy.deepcopy(dict(p14["conventional_unwhitened_adapt"])),
                    "points": adapt14,
                    "terminal": copy.deepcopy(adapt14[-1]),
                },
                "page14_proxy_beam": {
                    **copy.deepcopy(dict(p14["beam_metric_route"])),
                    "points": _prefix_points(
                        p14["beam_metric_route"], label="Page-14 proxy beam"
                    ),
                    "terminal": copy.deepcopy(
                        _prefix_points(
                            p14["beam_metric_route"], label="Page-14 proxy beam"
                        )[-1]
                    ),
                },
                "page16_unpruned_qiskit": {
                    **copy.deepcopy(dict(p16["page16_qiskit_route"])),
                    "points": _prefix_points(
                        p16["page16_qiskit_route"], label="Page-16 Qiskit"
                    ),
                    "terminal": copy.deepcopy(
                        _prefix_points(
                            p16["page16_qiskit_route"], label="Page-16 Qiskit"
                        )[-1]
                    ),
                    "costs": None,
                    "cost_scope": "not_reused_source_terminal_is_not_k20",
                },
                "page18_qiskit_beam_metric": route,
                "status": (
                    "completed_authenticated_chtc_archive"
                    if route is not None
                    else "pending_no_schema_locked_retrieval"
                ),
                "job": binding(job_path),
            }
        )
    completed_count = sum(
        row["page18_qiskit_beam_metric"] is not None for row in cells
    )
    unsigned: dict[str, Any] = {
        "schema": ADAPTER_SCHEMA,
        "page_id": PAGE_ID,
        "status": (
            "completed_6_of_6_exact_k20"
            if completed_count == 6
            else f"partial_{completed_count}_of_6_exact_k20"
        ),
        "run_class": "candidate",
        "paper_evidence_adopted": False,
        "cluster_id": CLUSTER_ID,
        "completed_regime_count": completed_count,
        "pending_regime_count": 6 - completed_count,
        "comparison_round": TARGET_HORIZON,
        "route": {
            "candidate_representation": "macro_generator_v1",
            "phase0": "standard_absolute_energy_gradient",
            "phase0_cap": 24,
            "phase1_cap": 24,
            "phase2_cap": 24,
            "phase3_cap": 24,
            "phase1_cost_source": "measurement_proxy",
            "phase2_cost_source": "signed_qiskit_compiled_marginal",
            "phase3_cost_source": "signed_qiskit_compiled_marginal",
            "lane_shortlisting": False,
            "beam": "fork_local_three_branches_keep_two_v1",
            "metric_pruning": "metric_pruning_v1",
            "gradient_policy": "stationary_source_response_v1",
            "insertion": "commutation_reduced_relative_plateau",
            "relative_plateau_threshold": 1.0e-4,
            "optimizer": "powell",
            "optimizer_maxiter": 200,
            "seed": 7,
            "target_horizon": TARGET_HORIZON,
        },
        "source_package": {
            "manifest": binding(PACKAGE_DIR / "package_manifest.json"),
            "canonical_sha256": PACKAGE_MANIFEST_CANONICAL_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        },
        "submission_receipt": binding(SUBMISSION_RECEIPT),
        "retrieval_manifest": (
            binding(RETRIEVAL_MANIFEST) if RETRIEVAL_MANIFEST.is_file() else None
        ),
        "source_page14_adapter": {
            **binding(PAGE14_ADAPTER),
            "canonical_sha256": PAGE14_ADAPTER_CANONICAL_SHA256,
        },
        "source_page16_adapter": {
            **binding(PAGE16_ADAPTER),
            "canonical_sha256": PAGE16_ADAPTER_CANONICAL_SHA256,
        },
        "cells": cells,
        "limitations": [
            "the page is a round-20 candidate comparison and is not a round-50 paper-facing resource replacement",
            "Page-16 source-terminal cost tuples are not reused because those source terminals are k=30 or k=50, not the common k=20 comparison round",
            "only Page-18 archives named by the self-digested verified-retrieval manifest can become completed curves",
            "no paper-evidence adoption is implied",
        ],
    }
    unsigned["sha256"] = _canonical_sha256(unsigned)
    return unsigned


def format_error(value: float) -> str:
    return f"{float(value):.2e}"


def format_s_alg(value: int) -> str:
    mantissa, exponent = f"{int(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_cost_tuple(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return "--"
    return "(" + ",".join(
        format_s_alg(int(value[field])) if field == "S_alg" else str(int(value[field]))
        for field in ("N2q", "D2q", "Dc", "W1q", "S_alg")
    ) + ")"


def render_page(adapter: Mapping[str, Any]) -> dict[str, Any]:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _atomic_json(ADAPTER_PATH, adapter)
    mpl.rcParams.update({"font.family": "serif", "font.size": 7.0})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.64),
        hspace=0.34,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    styles = (
        ("conventional_unwhitened_adapt", BLUE, 1.15),
        ("page14_proxy_beam", PURPLE, 1.35),
        ("page16_unpruned_qiskit", GREEN, 1.45),
    )
    for index, (axis, cell) in enumerate(zip(axes, adapter["cells"], strict=True)):
        for key, color, width in styles:
            points = cell[key]["points"]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=color,
                lw=width,
            )
        new = cell["page18_qiskit_beam_metric"]
        if new is not None:
            points = new["points"]
            axis.plot(
                [row["k"] for row in points],
                [max(float(row["error"]), PLOT_FLOOR) for row in points],
                color=ORANGE,
                lw=1.9,
            )
            axis.scatter(
                [20],
                [max(float(new["terminal"]["error"]), PLOT_FLOOR)],
                color=ORANGE,
                marker="D",
                s=28,
                zorder=5,
            )
        else:
            axis.text(
                0.5,
                0.11,
                "Qiskit beam + metric: pending",
                transform=axis.transAxes,
                ha="center",
                fontsize=6.4,
                color=ORANGE,
                bbox={"facecolor": "white", "edgecolor": ORANGE, "alpha": 0.86},
            )
        axis.set_yscale("log")
        axis.set_xlim(0, TARGET_HORIZON)
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{cell['regime_label']} ($n_{{ph}}={cell['nph']}$)", fontsize=8.1
        )
        if index // 3 == 1:
            axis.set_xlabel("ADAPT controller round")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E|$")
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.15, label="Conventional unwhitened ADAPT"),
            Line2D([0], [0], color=PURPLE, lw=1.35, label="Page 14: proxy beam + metric"),
            Line2D([0], [0], color=GREEN, lw=1.45, label="Page 16: unpruned Qiskit II/III"),
            Line2D([0], [0], color=ORANGE, lw=1.9, label="Page 18: Qiskit II/III + beam + metric"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Macro Phase 0/1/2/3 at round 20: Qiskit II/III with beam and metric pruning",
        fontsize=10.6,
        fontweight="bold",
        y=0.988,
    )
    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for cell in adapter["cells"]:
        new = cell["page18_qiskit_beam_metric"]
        rows.append(
            [
                cell["regime_label"],
                format_error(cell["conventional_unwhitened_adapt"]["terminal"]["error"]),
                format_error(cell["page14_proxy_beam"]["terminal"]["error"]),
                format_error(cell["page16_unpruned_qiskit"]["terminal"]["error"]),
                "--" if new is None else format_error(new["terminal"]["error"]),
                "--" if new is None else format_cost_tuple(new["costs"]),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            r"ADAPT $|\Delta E_{20}|$",
            r"Page-14 $|\Delta E_{20}|$",
            r"Page-16 $|\Delta E_{20}|$",
            r"Page-18 $|\Delta E_{20}|$",
            r"Page-18 $(N_{2q},D_{2q},D_c,W_{1q},S_{alg})$",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.14, 0.13, 0.14, 0.14, 0.14, 0.31),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    table.scale(1.0, 0.86)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAEA")
    fig.text(
        0.5,
        0.017,
        r"All trajectories end at $k=20$. Page-18 tuple uses the shared locked Table-I compiler; $S_{alg}$ uses X.YeZ notation.",
        ha="center",
        fontsize=6.2,
    )
    completed_pages._save_page(fig, png_path=PAGE_PNG, pdf_path=PAGE_PDF)
    plt.close(fig)
    standalone_unsigned: dict[str, Any] = {
        "schema": STANDALONE_SCHEMA,
        "page_id": PAGE_ID,
        "status": adapter["status"],
        "paper_evidence_adopted": False,
        "canonical_master_mutated": False,
        "adapter": {**binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
        "outputs": {
            "page_pdf": binding(PAGE_PDF),
            "page_png": binding(PAGE_PNG),
        },
        "source_master_observation": (
            binding(TARGET_PDF) if TARGET_PDF.is_file() else None
        ),
        "limitations": copy.deepcopy(adapter["limitations"]),
    }
    standalone_unsigned["sha256"] = _canonical_sha256(standalone_unsigned)
    _atomic_json(STANDALONE_PROVENANCE, standalone_unsigned)
    return standalone_unsigned


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    data = b"" if contents is None else contents.get_data()
    return hashlib.sha256(data).hexdigest()


def _active_watcher_pids() -> list[int]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command=", "-ww"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UpdateError("cannot prove Page-17 watcher absence") from exc
    own_pid = os.getpid()
    result = []
    for raw in output.splitlines():
        line = raw.strip()
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != own_pid and WATCHER_NAME in command:
            result.append(pid)
    return result


@contextmanager
def _exclusive_watcher_absence() -> Iterator[None]:
    if not WATCH_LOCK.is_file() or WATCH_LOCK.is_symlink():
        raise UpdateError("Page-17 watcher lock is absent or unsafe")
    with WATCH_LOCK.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise UpdateError("Page-17 watcher is active; canonical append refused") from exc
        if _active_watcher_pids():
            raise UpdateError("Page-17 watcher process is active; canonical append refused")
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def append_to_canonical(
    adapter: Mapping[str, Any], standalone: Mapping[str, Any]
) -> dict[str, Any]:
    from pypdf import PdfReader, PdfWriter

    if adapter.get("status") != "completed_6_of_6_exact_k20":
        raise UpdateError("canonical Page-18 append requires all six exact-k20 cells")
    verify_self_digest(standalone, label="Page-18 standalone provenance")
    if (
        standalone.get("page_id") != PAGE_ID
        or standalone.get("adapter", {}).get("canonical_sha256")
        != adapter.get("sha256")
        or standalone.get("outputs", {}).get("page_pdf") != binding(PAGE_PDF)
    ):
        raise UpdateError("standalone Page-18 provenance drifted")
    with _exclusive_watcher_absence():
        provenance = load(TARGET_PROVENANCE)
        current = binding(TARGET_PDF)
        layout = provenance.get("layout")
        declared = provenance.get("outputs", {}).get("partial_progress_pdf")
        if (
            not isinstance(layout, Mapping)
            or not isinstance(declared, Mapping)
            or current["sha256"] != declared.get("sha256")
            or current["size_bytes"] != declared.get("size_bytes")
            or layout.get("page_count") != 17
            or layout.get("page_16") != PAGE16_ID
            or layout.get("page_17") != PAGE17_ID
            or "page_18" in layout
        ):
            raise UpdateError("canonical Pages 1--17 provenance is unsupported")
        original = PdfReader(str(TARGET_PDF), strict=False)
        page18 = PdfReader(str(PAGE_PDF), strict=False)
        if len(original.pages) != 17 or len(page18.pages) != 1:
            raise UpdateError("canonical append requires exactly 17 + 1 pages")
        preserved_hashes = [_page_content_sha256(row) for row in original.pages]
        writer = PdfWriter()
        for row in original.pages:
            writer.add_page(row)
        writer.add_page(page18.pages[0])

        token = uuid.uuid4().hex
        temporary_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.tmp")
        temporary_provenance = TARGET_PROVENANCE.with_name(
            f".{TARGET_PROVENANCE.name}.{token}.tmp"
        )
        rollback_pdf = TARGET_PDF.with_name(f".{TARGET_PDF.name}.{token}.rollback")
        rollback_provenance = TARGET_PROVENANCE.with_name(
            f".{TARGET_PROVENANCE.name}.{token}.rollback"
        )
        try:
            with temporary_pdf.open("xb") as stream:
                writer.write(stream)
                stream.flush()
                os.fsync(stream.fileno())
            combined = PdfReader(str(temporary_pdf), strict=False)
            if len(combined.pages) != 18:
                raise UpdateError("combined canonical report is not 18 pages")
            if [
                _page_content_sha256(row) for row in combined.pages[:17]
            ] != preserved_hashes:
                raise UpdateError("canonical Page-18 append changed Pages 1--17")
            updated = copy.deepcopy(provenance)
            updated["layout"]["page_18"] = PAGE_ID
            updated["layout"]["page_count"] = 18
            updated["macro_phase23_qiskit_beam_metric_page18"] = {
                "schema": MASTER_REPORT_SCHEMA,
                "page_id": PAGE_ID,
                "status": adapter["status"],
                "paper_evidence_adopted": False,
                "adapter": {
                    **binding(ADAPTER_PATH),
                    "canonical_sha256": adapter["sha256"],
                },
                "cells": copy.deepcopy(adapter["cells"]),
                "limitations": copy.deepcopy(adapter["limitations"]),
                "preserved_page_content_sha256": preserved_hashes,
                "source_provenance_before_append": binding(TARGET_PROVENANCE),
                "outputs": {
                    "page_pdf": binding(PAGE_PDF),
                    "page_png": binding(PAGE_PNG),
                },
            }
            for key, path in (
                ("macro_phase23_qiskit_beam_metric_page18_pdf", PAGE_PDF),
                ("macro_phase23_qiskit_beam_metric_page18_png", PAGE_PNG),
                ("macro_phase23_qiskit_beam_metric_page18_adapter", ADAPTER_PATH),
                (
                    "macro_phase23_qiskit_beam_metric_page18_provenance",
                    STANDALONE_PROVENANCE,
                ),
            ):
                updated["outputs"][key] = binding(path)
            combined_binding = binding(temporary_pdf)
            combined_binding["path"] = str(TARGET_PDF.resolve())
            updated["outputs"]["partial_progress_pdf"] = combined_binding
            with temporary_provenance.open("xb") as stream:
                stream.write(
                    json.dumps(updated, indent=2, sort_keys=True, allow_nan=False).encode()
                    + b"\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.link(TARGET_PDF, rollback_pdf)
            os.link(TARGET_PROVENANCE, rollback_provenance)
            os.replace(temporary_pdf, TARGET_PDF)
            try:
                os.replace(temporary_provenance, TARGET_PROVENANCE)
            except BaseException:
                os.replace(rollback_pdf, TARGET_PDF)
                os.replace(rollback_provenance, TARGET_PROVENANCE)
                raise
            rollback_pdf.unlink(missing_ok=True)
            rollback_provenance.unlink(missing_ok=True)
        except BaseException:
            temporary_pdf.unlink(missing_ok=True)
            temporary_provenance.unlink(missing_ok=True)
            rollback_pdf.unlink(missing_ok=True)
            rollback_provenance.unlink(missing_ok=True)
            raise
    return {
        "status": "appended_page18_to_canonical",
        "page_count": 18,
        "preserved_page_count": 17,
        "pdf": binding(TARGET_PDF),
        "provenance": binding(TARGET_PROVENANCE),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--append-canonical",
        action="store_true",
        help="append only after six-cell closure and Page-17 watcher absence",
    )
    args = parser.parse_args(argv)
    try:
        adapter = build_adapter()
        standalone = render_page(adapter)
        result: dict[str, Any] = {
            "status": "standalone_page18_updated",
            "adapter_status": adapter["status"],
            "completed_cells": adapter["completed_regime_count"],
            "page_pdf": binding(PAGE_PDF),
            "page_png": binding(PAGE_PNG),
            "adapter": binding(ADAPTER_PATH),
            "provenance": binding(STANDALONE_PROVENANCE),
            "canonical_master_mutated": False,
        }
        if args.append_canonical:
            result["canonical_append"] = append_to_canonical(adapter, standalone)
            result["canonical_master_mutated"] = True
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, tarfile.TarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
