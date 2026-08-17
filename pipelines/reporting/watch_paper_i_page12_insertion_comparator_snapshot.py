#!/usr/bin/env python3
"""Append and refresh dense Page 18 from authenticated Page-12 receipts.

This is a credentials-free, reporting-only watcher.  It never connects to
CHTC, launches science, changes scheduler state, or treats a path as evidence.
The first Page-12 comparator curve is admitted only after the archive finalizer
has emitted a self-digested closure receipt and this module has independently
closed that receipt against the fixed package, job, archive, worker receipt,
execution manifest, and full tar-member inventory.

Page 17 remains the intact-macro insertion-comparator snapshot.  This module
adds exactly one dense 2-by-3 singleton Page 18 and replaces only that page as
new authenticated receipts arrive.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from typing import Any, Mapping
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PACKAGE_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
PACKAGE_MANIFEST = PACKAGE_DIR / "package_manifest.json"
FINALIZER_PATH = PACKAGE_ROOT / (
    "finalize_page12_insertion_comparator_closure_20260813.py"
)
EXPECTED_FINALIZER_SHA256 = (
    "b9362052044aa85861e7046bc248cf6ed96c4949183059fb8d64831fbb8daeb6"
)
EXPECTED_ACTIVATION_SHA256 = (
    "9aa36c3362257dfdcd8624bf091adfbaae28edb06e0abadcb8d6b6936533a36d"
)
ACTIVATION_DIR = PACKAGE_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_"
    "20260812_v1_chtc_activation_v1"
)
RECEIPT_DIR = PACKAGE_ROOT / "page12_insertion_comparator_closure_receipts"
RETRIEVED_DIR = PACKAGE_ROOT / "retrieved_page12_insertion_comparators_20260813"
IDENTITY_DIR = PACKAGE_ROOT / "page12_insertion_comparator_closure_evidence"

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
REFERENCE_ADAPTER = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "global_singleton_gradient_phase0_page12_adapter.json"
)
STEM = (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "page12_singleton_insertion_comparator_snapshot"
)
PAGE18_PDF = REPORT_DIR / f"{STEM}_page18.pdf"
PAGE18_PNG = REPORT_DIR / f"{STEM}_page18.png"
ADAPTER_PATH = REPORT_DIR / f"{STEM}_adapter.json"
WATCH_STATUS_PATH = REPORT_DIR / f"{STEM}_watch_status.json"
LOCK_PATH = REPORT_DIR / f"{STEM}_watch.lock"
REPORT_MUTATION_LOCK_PATH = REPORT_DIR / (
    "paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_"
    "partial_progress_mutation.lock"
)

# This is the current dense macro-page identity.  It is deliberately repeated
# instead of importing the v2 macro campaign implementation into this watcher.
PAGE17_ID = "phase0_insertion_comparator_page16_six_regime_progress_snapshot_v3"
PAGE18_ID = "phase0_insertion_comparator_page12_six_regime_progress_snapshot_v1"
PAGE19_ID = "l3_weak_holstein_page12_vs_append_k50_progress_v1"
PAGE12_REFERENCE_ID = "global_singleton_gradient_phase0_partial_v1"

RECEIPT_SCHEMA = "paper_i_ra_adapt_page12_insertion_comparator_closure_receipt_v1"
RECEIPT_STATUS = "passed_authenticated_page12_insertion_comparator_closure"
ADAPTER_SCHEMA = "paper_i_ra_adapt_page12_insertion_comparator_progress_adapter_v1"
STATUS_SCHEMA = "paper_i_page12_insertion_comparator_page18_auto_refresh_status_v1"
CLUSTER_ID = 9647385
TARGET_HORIZON = 50
MIN_POLL_SECONDS = 30.0
DEFAULT_MAX_POLL_SECONDS = 300.0
MAX_CAPTURE_MEMBER_BYTES = 32 * 1024 * 1024

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
EXPECTED_POLICIES = ("always_commutation_reduced", "append_only")
POLICY_DISPLAY = {
    EXPECTED_POLICIES[0]: "RA-ADAPT insertion always",
    EXPECTED_POLICIES[1]: "RA-ADAPT append-only insertion (append always)",
}
PLOT_FLOOR = 1.0e-16
BLUE = "#4C78A8"
GREEN = "#009E73"
ORANGE = "#E69F00"
MAGENTA = "#CC79A7"
GRAY = "#666666"
RED = "#B22222"


class WatchError(ValueError):
    """A receipt, source, or report failed closed authentication."""


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WatchError(f"cannot load JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise WatchError(f"JSON object required: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    if not _is_sha256(claimed) or claimed != _canonical_sha256(unsigned):
        raise WatchError(f"{label}: self digest drifted")


def binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise WatchError(f"unsafe or missing file: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


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


def _safe_member_name(raw: Any, *, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise WatchError(f"{label}: archive member path is absent")
    normalized = raw
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WatchError(f"{label}: unsafe archive member path {raw!r}")
    return path.as_posix()


def _resolve_repo_or_absolute(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise WatchError(f"{label}: path is absent")
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def receipt_filename(proc_id: int, run_id: str) -> str:
    return (
        f"paper_i_ra_adapt_page12_cluster{CLUSTER_ID}_proc{proc_id:02d}_"
        f"{run_id}_closure_receipt_20260813.json"
    )


def _load_package() -> tuple[dict[str, Any], dict[str, tuple[Path, dict[str, Any]]]]:
    if not PACKAGE_MANIFEST.is_file() or PACKAGE_MANIFEST.is_symlink():
        raise WatchError("fixed Page-12 package manifest is absent or unsafe")
    package = load(PACKAGE_MANIFEST)
    _verify_self_digest(package, label="Page-12 comparator package")
    rows = package.get("jobs")
    if (
        package.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_package_manifest_v1"
        or package.get("status") != "passed_inert_twelve_cells"
        or package.get("row_count") != 12
        or package.get("weak_holstein_horizon") != TARGET_HORIZON
        or package.get("strong_holstein_horizon") != TARGET_HORIZON
        or tuple(package.get("comparator_policies", ())) != EXPECTED_POLICIES
        or package.get("plateau_reference_reused_not_rerun") is not True
        or not isinstance(rows, list)
        or len(rows) != 12
    ):
        raise WatchError("fixed Page-12 package identity drifted")
    jobs: dict[str, tuple[Path, dict[str, Any]]] = {}
    cells: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise WatchError("Page-12 package job binding is malformed")
        relative = _safe_member_name(row.get("path"), label="package job")
        path = PACKAGE_DIR / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256_file(path) != row.get("sha256")
        ):
            raise WatchError(f"Page-12 package job file drifted: {relative}")
        job = load(path)
        _verify_self_digest(job, label=f"Page-12 job {relative}")
        execution_id = job.get("execution_id")
        regime = job.get("regime_id")
        policy = job.get("comparator_policy")
        if (
            not isinstance(execution_id, str)
            or execution_id in jobs
            or row.get("execution_id") != execution_id
            or row.get("canonical_sha256") != job.get("sha256")
            or regime not in REGIME_ORDER
            or policy not in EXPECTED_POLICIES
            or (regime, policy) in cells
            or job.get("target_horizon") != TARGET_HORIZON
            or job.get("candidate_representation") != "single_pauli_word_v1"
            or job.get("typed_insertion_kind") != policy
            or job.get("runtime_insertion_mode")
            != (
                "full_commutation_reduced"
                if policy == EXPECTED_POLICIES[0]
                else "append_only"
            )
            or not _is_sha256(job.get("protocol_sha256"))
            or not _is_sha256(job.get("route_contract_sha256"))
        ):
            raise WatchError(f"Page-12 package job identity drifted: {relative}")
        jobs[execution_id] = (path, job)
        cells.add((str(regime), str(policy)))
    expected_cells = {
        (regime, policy) for regime in REGIME_ORDER for policy in EXPECTED_POLICIES
    }
    if cells != expected_cells:
        raise WatchError("Page-12 package does not close the six-by-two matrix")
    return package, jobs


def _authorized_job(run_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    package, jobs = _load_package()
    try:
        path, job = jobs[run_id]
    except KeyError as exc:
        raise WatchError("receipt run ID is not in the fixed Page-12 package") from exc
    return path, job, package


def _binding_shape(value: Any, *, label: str, canonical: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WatchError(f"{label}: binding is absent")
    if (
        not _is_sha256(value.get("sha256"))
        or not isinstance(value.get("size_bytes"), int)
        or int(value.get("size_bytes", -1)) <= 0
    ):
        raise WatchError(f"{label}: file binding is malformed")
    if canonical and not _is_sha256(value.get("canonical_sha256")):
        raise WatchError(f"{label}: canonical binding is malformed")
    return value


def _verify_local_binding(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    canonical: bool = True,
) -> tuple[Path, dict[str, Any] | None]:
    row = _binding_shape(value, label=label, canonical=canonical)
    path = _resolve_repo_or_absolute(row.get("path"), label=label)
    if (
        (expected_path is not None and path.resolve() != expected_path.resolve())
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != row.get("size_bytes")
        or _sha256_file(path) != row.get("sha256")
    ):
        raise WatchError(f"{label}: local file binding drifted")
    value_object = None
    if canonical:
        value_object = load(path)
        _verify_self_digest(value_object, label=label)
        if row.get("canonical_sha256") != value_object.get("sha256"):
            raise WatchError(f"{label}: canonical local binding drifted")
    return path, value_object


def _inside_archive_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path_inside_archive"), str)
        or not _is_sha256(value.get("canonical_sha256"))
        or not isinstance(value.get("schema"), str)
        or value.get("status") != "passed"
    ):
        raise WatchError(f"{label}: archived canonical binding is malformed")
    return value


def _stream_archive(
    archive_path: Path,
    *,
    inventory_rows: Any,
    capture_names: set[str],
) -> dict[str, bytes]:
    if not isinstance(inventory_rows, list) or not inventory_rows:
        raise WatchError("finalizer archive inventory is absent")
    inventory: dict[str, Mapping[str, Any]] = {}
    for row in inventory_rows:
        if not isinstance(row, Mapping):
            raise WatchError("finalizer archive inventory row is malformed")
        name = _safe_member_name(row.get("path"), label="receipt inventory")
        if (
            name in inventory
            or not _is_sha256(row.get("sha256"))
            or not isinstance(row.get("size_bytes"), int)
            or int(row.get("size_bytes", -1)) < 0
        ):
            raise WatchError("finalizer archive inventory is not unique and closed")
        inventory[name] = row
    captured: dict[str, bytes] = {}
    observed: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                if member.isdir():
                    # Directory headers carry no evidence.  Root and ordinary
                    # directories are allowed, but links and special files are not.
                    continue
                name = _safe_member_name(member.name, label="archive")
                row = inventory.get(name)
                if (
                    row is None
                    or name in observed
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != row.get("size_bytes")
                ):
                    raise WatchError(f"archive member is unsafe or unbound: {name}")
                source = archive.extractfile(member)
                if source is None:
                    raise WatchError(f"archive member is unreadable: {name}")
                digest = hashlib.sha256()
                size = 0
                capture = io.BytesIO() if name in capture_names else None
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    if capture is not None:
                        if size > MAX_CAPTURE_MEMBER_BYTES:
                            raise WatchError(f"captured archive member is too large: {name}")
                        capture.write(block)
                if size != member.size or digest.hexdigest() != row.get("sha256"):
                    raise WatchError(f"archive member hash/size drifted: {name}")
                observed.add(name)
                if capture is not None:
                    captured[name] = capture.getvalue()
    except (OSError, tarfile.TarError) as exc:
        raise WatchError(f"cannot stream authenticated archive: {archive_path}") from exc
    if observed != set(inventory):
        raise WatchError("archive regular-member inventory is incomplete")
    if set(captured) != capture_names:
        raise WatchError("required reporting members are absent from archive")
    return captured


def _json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WatchError(f"{label}: member is not a JSON object") from exc
    if not isinstance(value, dict):
        raise WatchError(f"{label}: member is not a JSON object")
    return value


def _summary_result(
    summary: Mapping[str, Any], *, run_id: str, controller_rounds: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
    provenance = summary.get("provenance")
    trace = summary.get("accepted_error_trace")
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or not isinstance(provenance, Mapping)
        or not isinstance(trace, list)
        or len(trace) != controller_rounds
        or [row.get("controller_round") for row in trace if isinstance(row, Mapping)]
        != list(range(1, controller_rounds + 1))
    ):
        raise WatchError(f"authenticated summary shape drifted: {run_id}")
    exact = provenance.get("exact_same_cutoff_energy")
    if not isinstance(exact, (int, float)) or not math.isfinite(float(exact)):
        raise WatchError(f"same-cutoff reference is invalid: {run_id}")
    points: list[dict[str, Any]] = []
    for raw in trace:
        if not isinstance(raw, Mapping):
            raise WatchError(f"accepted trajectory row is malformed: {run_id}")
        k = raw.get("controller_round")
        energy = raw.get("accepted_energy")
        error = raw.get("absolute_energy_error")
        depth = raw.get("active_ansatz_depth")
        if (
            not isinstance(k, int)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or float(error) < 0.0
            or not math.isclose(
                abs(float(energy) - float(exact)),
                float(error),
                rel_tol=0.0,
                abs_tol=2.0e-10,
            )
            or not isinstance(depth, int)
            or depth < 0
        ):
            raise WatchError(f"accepted trajectory row is invalid: {run_id}/k={k}")
        points.append(
            {
                "k": k,
                "energy": float(energy),
                "error": float(error),
                "active_ansatz_depth": depth,
            }
        )
    plateau = summary.get("effective_plateau")
    marker_k = controller_rounds
    marker_policy = "terminal_plotted_point"
    if isinstance(plateau, Mapping):
        candidate = plateau.get("controller_round")
        if isinstance(candidate, int) and 1 <= candidate <= controller_rounds:
            marker_k = candidate
            marker_policy = "first_effective_plateau_prefix"
    marker_point = points[marker_k - 1]
    marker = {
        "k": marker_k,
        "error": marker_point["error"],
        "policy": marker_policy,
    }
    terminal = {"k": controller_rounds, "error": points[-1]["error"]}
    return points, terminal, marker, float(exact)


def authenticate_receipt(path: Path) -> dict[str, Any]:
    """Independently authenticate one exact finalizer receipt and its curve."""

    if (
        not FINALIZER_PATH.is_file()
        or FINALIZER_PATH.is_symlink()
        or _sha256_file(FINALIZER_PATH) != EXPECTED_FINALIZER_SHA256
    ):
        raise WatchError("pinned Page-12 closure finalizer source drifted")
    if not path.is_file() or path.is_symlink():
        raise WatchError(f"closure receipt is absent or unsafe: {path}")
    receipt = load(path)
    _verify_self_digest(receipt, label="Page-12 closure receipt")
    checks = receipt.get("authentication_checks")
    required_top = (
        "activation_manifest",
        "authorization",
        "remote_local_identity_evidence",
        "package_manifest",
        "job",
        "protocol",
        "archive",
        "worker_receipt",
        "execution_manifest",
        "summary_json",
    )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != RECEIPT_STATUS
        or receipt.get("cluster_id") != CLUSTER_ID
        or not isinstance(receipt.get("proc_id"), int)
        or not 0 <= int(receipt.get("proc_id", -1)) < 12
        or not isinstance(receipt.get("run_id"), str)
        or receipt.get("regime_id") not in REGIME_ORDER
        or receipt.get("comparator_policy") not in EXPECTED_POLICIES
        or not isinstance(receipt.get("controller_rounds_completed"), int)
        or not 1 <= int(receipt.get("controller_rounds_completed", -1)) <= TARGET_HORIZON
        or any(key not in receipt for key in required_top)
        or not isinstance(checks, Mapping)
        or len(checks) < 4
        or any(value is not True for value in checks.values())
    ):
        raise WatchError("closure receipt lacks finalizer authentication contract")
    run_id = str(receipt["run_id"])
    proc_id = int(receipt["proc_id"])
    if path.name != receipt_filename(proc_id, run_id):
        raise WatchError("closure receipt filename does not bind its identity")
    _job_path, job, package = _authorized_job(run_id)
    expected_proc = REGIME_ORDER.index(str(job["regime_id"]))
    if job["comparator_policy"] == EXPECTED_POLICIES[1]:
        expected_proc += len(REGIME_ORDER)
    if (
        receipt.get("regime_id") != job.get("regime_id")
        or receipt.get("comparator_policy") != job.get("comparator_policy")
        or proc_id != expected_proc
        or job.get("target_horizon") != TARGET_HORIZON
    ):
        raise WatchError("closure receipt does not bind the authorized queue cell")

    package_binding = _binding_shape(
        receipt["package_manifest"], label="receipt package", canonical=True
    )
    job_binding = _binding_shape(receipt["job"], label="receipt job", canonical=True)
    protocol_binding = _binding_shape(
        receipt["protocol"], label="receipt protocol", canonical=True
    )
    activation_binding = receipt["activation_manifest"]
    authorization_binding = receipt["authorization"]
    if (
        package_binding.get("canonical_sha256") != package.get("sha256")
        or job_binding.get("canonical_sha256") != job.get("sha256")
        or protocol_binding.get("canonical_sha256") != job.get("protocol_sha256")
        or receipt.get("route_contract_canonical_sha256")
        != job.get("route_contract_sha256")
        or receipt.get("typed_insertion_kind") != job.get("typed_insertion_kind")
        or receipt.get("runtime_insertion_mode") != job.get("runtime_insertion_mode")
        or not isinstance(activation_binding, Mapping)
        or not _is_sha256(activation_binding.get("canonical_sha256"))
        or not isinstance(authorization_binding, Mapping)
        or not _is_sha256(authorization_binding.get("canonical_sha256"))
    ):
        raise WatchError("closure receipt package/activation authority drifted")
    protocol_path = PACKAGE_DIR / str(job["protocol_path"])
    _verify_local_binding(
        package_binding,
        label="receipt package",
        expected_path=PACKAGE_MANIFEST,
    )
    _verify_local_binding(
        job_binding,
        label="receipt job",
        expected_path=_job_path,
    )
    _verify_local_binding(
        protocol_binding,
        label="receipt protocol",
        expected_path=protocol_path,
    )
    activation_path, activation_value = _verify_local_binding(
        activation_binding,
        label="receipt activation",
        expected_path=ACTIVATION_DIR / "activation_manifest.json",
    )
    authorization_path, authorization_value = _verify_local_binding(
        authorization_binding,
        label="receipt authorization",
        expected_path=ACTIVATION_DIR / "authorizations" / f"{run_id}.json",
    )
    if (
        activation_path.parent != ACTIVATION_DIR
        or authorization_path.parent != ACTIVATION_DIR / "authorizations"
        or not isinstance(activation_value, Mapping)
        or activation_value.get("sha256") != EXPECTED_ACTIVATION_SHA256
        or activation_value.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_activation_manifest_v1"
        or activation_value.get("status")
        != "passed_activation_prepared_no_submission"
        or activation_value.get("package_manifest_sha256") != package.get("sha256")
        or activation_value.get("authorization_count") != 12
        or activation_value.get("execution_authorized") is not True
        or activation_value.get("submission_authorized") is not True
        or activation_value.get("paper_evidence_adoption_authorized") is not False
        or activation_value.get("submitted") is not False
        or not isinstance(authorization_value, Mapping)
        or authorization_value.get("schema")
        != (
            "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
            "execution_authorization_v1"
        )
        or authorization_value.get("authorization_kind")
        != "explicit_user_execution_and_submission_authority"
        or authorization_value.get("scope") != "single_cell_chtc_execution_only"
        or authorization_value.get("execution_id") != run_id
        or authorization_value.get("job_spec_sha256") != job.get("sha256")
        or authorization_value.get("protocol_sha256") != job.get("protocol_sha256")
        or authorization_value.get("package_manifest_sha256") != package.get("sha256")
        or authorization_value.get("execution_authorized") is not True
        or authorization_value.get("submission_authorized") is not True
        or authorization_value.get("paper_evidence_adoption_authorized") is not False
        or authorization_value.get("submitted") is not False
    ):
        raise WatchError("closure receipt activation/authorization binding drifted")

    identity_path, identity_value = _verify_local_binding(
        receipt["remote_local_identity_evidence"],
        label="receipt remote/local identity",
        expected_path=(
            IDENTITY_DIR / f"{run_id}__{CLUSTER_ID}__{proc_id}_remote_archive_identity.json"
        ),
    )
    if (
        identity_path.parent != IDENTITY_DIR
        or not isinstance(identity_value, Mapping)
        or identity_value.get("schema")
        != (
            "paper_i_ra_adapt_page12_insertion_comparator_"
            "remote_archive_identity_v1"
        )
        or identity_value.get("status")
        != "passed_remote_local_size_sha256_match_after_atomic_rename"
        or identity_value.get("cluster_id") != CLUSTER_ID
        or identity_value.get("proc_id") != proc_id
        or identity_value.get("execution_id") != run_id
        or identity_value.get("gzip_integrity_passed") is not True
        or identity_value.get("tar_readability_passed") is not True
        or identity_value.get("atomic_local_rename_completed") is not True
    ):
        raise WatchError("closure receipt remote/local identity binding drifted")

    archive = _binding_shape(receipt["archive"], label="receipt archive")
    expected_archive = RETRIEVED_DIR / f"{run_id}__{CLUSTER_ID}__{proc_id}.tar.gz"
    archive_path = _resolve_repo_or_absolute(archive.get("path"), label="archive")
    remote_path = archive.get("remote_path")
    if (
        archive_path.resolve() != expected_archive.resolve()
        or archive_path.is_symlink()
        or not archive_path.is_file()
        or archive_path.stat().st_size != archive.get("size_bytes")
        or _sha256_file(archive_path) != archive.get("sha256")
        or not isinstance(remote_path, str)
        or "/outputs/transfer/" not in remote_path
        or not remote_path.endswith(expected_archive.name)
        or identity_value.get("remote_path") != remote_path
        or identity_value.get("local_path") != archive.get("path")
        or identity_value.get("remote_size_bytes") != archive.get("size_bytes")
        or identity_value.get("local_size_bytes") != archive.get("size_bytes")
        or identity_value.get("remote_sha256") != archive.get("sha256")
        or identity_value.get("local_sha256") != archive.get("sha256")
    ):
        raise WatchError("closure receipt archive identity drifted")
    run_root = f"runs/{run_id}"
    summary_name = f"{run_root}/summary/summary.json"
    manifest_name = f"{run_root}/execution_manifest.json"
    worker_name = "worker_receipt.json"
    exit_name = "worker_exit_status.txt"
    captured = _stream_archive(
        archive_path,
        inventory_rows=archive.get("inventory"),
        capture_names={summary_name, manifest_name, worker_name, exit_name},
    )
    if captured[exit_name].strip() != b"0":
        raise WatchError("authenticated worker exit status is not zero")

    worker_binding = _inside_archive_binding(
        receipt["worker_receipt"], label="receipt worker"
    )
    manifest_binding = _inside_archive_binding(
        receipt["execution_manifest"], label="receipt execution manifest"
    )
    inventory = {
        str(row["path"]): row
        for row in archive["inventory"]
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if (
        worker_binding.get("path_inside_archive") != worker_name
        or manifest_binding.get("path_inside_archive") != manifest_name
        or worker_name not in inventory
        or manifest_name not in inventory
    ):
        raise WatchError("closure receipt primary-member bindings drifted")
    summary_binding = receipt["summary_json"]
    if (
        not isinstance(summary_binding, Mapping)
        or summary_binding.get("path_inside_archive") != summary_name
        or summary_binding.get("sha256") != inventory[summary_name].get("sha256")
        or summary_binding.get("size_bytes") != inventory[summary_name].get("size_bytes")
    ):
        raise WatchError("closure receipt summary binding drifted")
    worker = _json_bytes(captured[worker_name], label="worker receipt")
    manifest = _json_bytes(captured[manifest_name], label="execution manifest")
    _verify_self_digest(worker, label="archived worker receipt")
    _verify_self_digest(manifest, label="archived execution manifest")
    if (
        worker_binding.get("canonical_sha256") != worker.get("sha256")
        or manifest_binding.get("canonical_sha256") != manifest.get("sha256")
        or worker.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_worker_receipt_v1"
        or worker.get("status") != "passed"
        or worker.get("execution_id") != run_id
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("authorization_sha256")
        != authorization_binding.get("canonical_sha256")
        or worker.get("execution_manifest_sha256") != manifest.get("sha256")
        or worker.get("controller_rounds_completed")
        != receipt.get("controller_rounds_completed")
        or worker.get("fresh_start") is not True
        or manifest.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_execution_manifest_v1"
        or manifest.get("status") != "passed"
        or manifest.get("execution_id") != run_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("authorization_sha256")
        != authorization_binding.get("canonical_sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256") != job.get("route_contract_sha256")
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or manifest.get("controller_rounds_completed")
        != receipt.get("controller_rounds_completed")
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
    ):
        raise WatchError("archived worker/execution authority drifted")
    worker_artifacts = worker.get("artifacts")
    if not isinstance(worker_artifacts, list):
        raise WatchError("archived worker artifact inventory is absent")
    declared_artifacts: set[str] = set()
    for row in worker_artifacts:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or row.get("path") in declared_artifacts
            or row.get("path") not in inventory
        ):
            raise WatchError("archived worker artifact inventory is unbound")
        declared_artifacts.add(str(row["path"]))
        bound = inventory[str(row["path"])]
        if any(row.get(key) != bound.get(key) for key in ("sha256", "size_bytes")):
            raise WatchError("archived worker artifact hash/size drifted")
    if declared_artifacts != set(inventory).difference({worker_name, exit_name}):
        raise WatchError("archived worker artifact inventory is not exact")
    output_payloads = manifest.get("output_payloads")
    expected_run_artifacts = job.get("expected_run_artifacts")
    if not isinstance(output_payloads, Mapping) or not isinstance(
        expected_run_artifacts, Mapping
    ):
        raise WatchError("archived execution output inventory is absent")
    expected_roles = {"checkpoint", "estimator_ledger", "result", "summary"}
    if set(output_payloads) != expected_roles:
        raise WatchError("archived execution output role inventory drifted")
    for role in expected_roles:
        expected = expected_run_artifacts.get(role)
        row = output_payloads.get(role)
        if not isinstance(expected, Mapping) or not isinstance(row, Mapping):
            raise WatchError(f"archived execution output is absent: {role}")
        relative = expected.get("path")
        if (
            not isinstance(relative, str)
            or relative not in inventory
            or row.get("path") != relative
            or row.get("sha256") != inventory[relative].get("sha256")
            or row.get("size_bytes") != inventory[relative].get("size_bytes")
        ):
            raise WatchError(f"archived execution output binding drifted: {role}")

    summary = _json_bytes(captured[summary_name], label="Page-12 summary")
    points, terminal, marker, exact_same_cutoff_energy = _summary_result(
        summary,
        run_id=run_id,
        controller_rounds=int(receipt["controller_rounds_completed"]),
    )
    return {
        "run_id": run_id,
        "cluster_id": CLUSTER_ID,
        "proc_id": proc_id,
        "regime_id": str(job["regime_id"]),
        "comparator_policy": str(job["comparator_policy"]),
        "controller_rounds_completed": int(receipt["controller_rounds_completed"]),
        "exact_same_cutoff_energy": exact_same_cutoff_energy,
        "points": points,
        "terminal": terminal,
        "marker": marker,
        "full_source_horizon": int(points[-1]["k"]),
        "plotted_horizon": int(points[-1]["k"]),
        "full_source_point_count": len(points),
        "plotted_point_count": len(points),
        "display_crop": "common_comparator_horizon_k_le_50",
        "receipt_sha256": str(receipt["sha256"]),
        "source": {
            "closure_receipt": {
                **binding(path),
                "canonical_sha256": receipt["sha256"],
            },
            "archive": {
                "path": str(archive_path.resolve()),
                "remote_path": remote_path,
                "sha256": archive["sha256"],
                "size_bytes": archive["size_bytes"],
                "member_count": len(inventory),
                "full_regular_member_inventory_closed": True,
            },
            "summary_member": {
                "path": summary_name,
                "sha256": inventory[summary_name]["sha256"],
                "size_bytes": inventory[summary_name]["size_bytes"],
            },
            "worker_receipt_canonical_sha256": worker["sha256"],
            "execution_manifest_canonical_sha256": manifest["sha256"],
            "job_canonical_sha256": job["sha256"],
            "package_manifest_canonical_sha256": package["sha256"],
        },
    }


def authenticated_inventory(receipt_dir: Path = RECEIPT_DIR) -> list[dict[str, Any]]:
    if not receipt_dir.exists():
        return []
    if receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise WatchError("Page-12 closure receipt directory is unsafe")
    paths = sorted(receipt_dir.glob("*_closure_receipt_20260813.json"))
    results = [authenticate_receipt(path) for path in paths]
    cells: set[tuple[str, str]] = set()
    procs: set[int] = set()
    for result in results:
        cell = (str(result["regime_id"]), str(result["comparator_policy"]))
        proc = int(result["proc_id"])
        if cell in cells or proc in procs:
            raise WatchError("duplicate Page-12 authenticated closure receipt")
        cells.add(cell)
        procs.add(proc)
    return sorted(results, key=lambda row: int(row["proc_id"]))


def _load_reference_adapter() -> dict[str, Any]:
    if not REFERENCE_ADAPTER.is_file() or REFERENCE_ADAPTER.is_symlink():
        raise WatchError("Page-12 plateau reference adapter is absent or unsafe")
    adapter = load(REFERENCE_ADAPTER)
    _verify_self_digest(adapter, label="Page-12 plateau reference adapter")
    cells = adapter.get("cells")
    if (
        adapter.get("schema") != "paper_i_phase0_route_progress_adapter_v1"
        or adapter.get("status") != "completed_six_regime_evidence_ready"
        or adapter.get("page_id") != PAGE12_REFERENCE_ID
        or adapter.get("route_key") != "global_singleton_gradient_phase0"
        or not isinstance(cells, list)
        or [row.get("regime_id") for row in cells if isinstance(row, Mapping)]
        != list(REGIME_ORDER)
    ):
        raise WatchError("Page-12 plateau reference identity drifted")
    return adapter


def build_adapter(results: list[dict[str, Any]]) -> dict[str, Any]:
    reference_adapter = _load_reference_adapter()
    completed: dict[str, dict[str, dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for result in results:
        regime = str(result["regime_id"])
        policy = str(result["comparator_policy"])
        if regime not in REGIME_ORDER or policy not in EXPECTED_POLICIES:
            raise WatchError("authenticated result is outside the Page-12 matrix")
        if (regime, policy) in seen:
            raise WatchError("authenticated Page-12 matrix cell is duplicated")
        seen.add((regime, policy))
        completed.setdefault(regime, {})[policy] = copy.deepcopy(result)

    references: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    reference_limitations: list[str] = []
    for cell in reference_adapter["cells"]:
        regime = str(cell["regime_id"])
        exact = cell.get("exact_same_cutoff_energy")
        route = cell.get("phase0_route")
        current_adapt = cell.get("append_adapt")
        points = route.get("points") if isinstance(route, Mapping) else None
        if (
            not isinstance(exact, (int, float))
            or not math.isfinite(float(exact))
            or not isinstance(route, Mapping)
            or not str(route.get("status", "")).startswith("completed_authenticated")
            or not isinstance(route.get("source"), Mapping)
            or not isinstance(points, list)
            or not points
            or [row.get("k") for row in points if isinstance(row, Mapping)]
            != list(range(1, int(points[-1].get("k", -1)) + 1))
            or int(points[-1].get("k", -1)) < TARGET_HORIZON
        ):
            raise WatchError(f"Page-12 plateau reference is incomplete: {regime}")
        normalized_full: list[dict[str, Any]] = []
        for point in points:
            error = point.get("error")
            energy = point.get("energy")
            if (
                not isinstance(error, (int, float))
                or not math.isfinite(float(error))
                or float(error) < 0.0
                or not isinstance(energy, (int, float))
                or not math.isfinite(float(energy))
                or not math.isclose(
                    abs(float(energy) - float(exact)),
                    float(error),
                    rel_tol=0.0,
                    abs_tol=2.0e-10,
                )
            ):
                raise WatchError(f"Page-12 reference error is invalid: {regime}")
            normalized_full.append(
                {"k": int(point["k"]), "error": float(error)}
            )
        normalized = [
            point for point in normalized_full if int(point["k"]) <= TARGET_HORIZON
        ]
        if [point["k"] for point in normalized] != list(
            range(1, TARGET_HORIZON + 1)
        ):
            raise WatchError(f"Page-12 plateau crop is incomplete: {regime}")
        terminal = normalized[-1]

        adapt_points = (
            current_adapt.get("points")
            if isinstance(current_adapt, Mapping)
            else None
        )
        adapt_marker = (
            current_adapt.get("marker")
            if isinstance(current_adapt, Mapping)
            else None
        )
        adapt_exact = (
            current_adapt.get("exact_same_cutoff_energy")
            if isinstance(current_adapt, Mapping)
            else None
        )
        if (
            not isinstance(adapt_exact, (int, float))
            or not math.isclose(
                float(adapt_exact),
                float(exact),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise WatchError(
                f"Page-12 current ADAPT same-cutoff reference drifted: {regime}"
            )
        if (
            not isinstance(current_adapt, Mapping)
            or not isinstance(current_adapt.get("execution_id"), str)
            or not current_adapt["execution_id"]
            or not isinstance(current_adapt.get("source"), Mapping)
            or not isinstance(adapt_points, list)
            or not adapt_points
            or [row.get("k") for row in adapt_points if isinstance(row, Mapping)]
            != list(range(0, int(adapt_points[-1].get("k", -1)) + 1))
            or int(adapt_points[-1].get("k", -1)) < TARGET_HORIZON
            or not isinstance(adapt_marker, Mapping)
            or adapt_marker.get("policy") != "terminal_common_horizon"
            or adapt_marker.get("k") != TARGET_HORIZON
        ):
            raise WatchError(f"Page-12 current ADAPT baseline is incomplete: {regime}")
        normalized_adapt_full: list[dict[str, Any]] = []
        for point in adapt_points:
            error = point.get("error")
            if (
                not isinstance(error, (int, float))
                or not math.isfinite(float(error))
                or float(error) < 0.0
            ):
                raise WatchError(
                    f"Page-12 current ADAPT error is invalid: {regime}"
                )
            normalized_adapt_full.append(
                {"k": int(point["k"]), "error": float(error)}
            )
        normalized_adapt = [
            point
            for point in normalized_adapt_full
            if int(point["k"]) <= TARGET_HORIZON
        ]
        if [point["k"] for point in normalized_adapt] != list(
            range(0, TARGET_HORIZON + 1)
        ):
            raise WatchError(f"Page-12 current ADAPT crop is incomplete: {regime}")
        marker_point = normalized_adapt[TARGET_HORIZON]
        marker_error = adapt_marker.get("error")
        if (
            not isinstance(marker_error, (int, float))
            or not math.isfinite(float(marker_error))
            or not math.isclose(
                float(marker_error),
                float(marker_point["error"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        ):
            raise WatchError(f"Page-12 current ADAPT marker drifted: {regime}")
        source_limitations = current_adapt["source"].get("limitations", [])
        if not isinstance(source_limitations, list) or any(
            not isinstance(value, str) for value in source_limitations
        ):
            raise WatchError(
                f"Page-12 current ADAPT source limitations are malformed: {regime}"
            )
        reference_limitations.extend(
            f"{REGIME_LABELS[regime]} current Append-ADAPT source: {value}"
            for value in source_limitations
        )
        for result in completed.get(regime, {}).values():
            result_exact = result.get("exact_same_cutoff_energy")
            if (
                not isinstance(result_exact, (int, float))
                or not math.isclose(
                    float(result_exact),
                    float(exact),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise WatchError(
                    f"Page-12 same-cutoff reference drifted: {regime}"
                )
        references.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": int(cell["nph"]),
                "exact_same_cutoff_energy": float(exact),
                "points": normalized,
                "terminal": copy.deepcopy(terminal),
                "marker": {
                    **copy.deepcopy(terminal),
                    "policy": "terminal_plotted_point",
                },
                "full_source_horizon": int(normalized_full[-1]["k"]),
                "plotted_horizon": TARGET_HORIZON,
                "full_source_point_count": len(normalized_full),
                "plotted_point_count": len(normalized),
                "display_crop": "common_comparator_horizon_k_le_50",
                "source": copy.deepcopy(route["source"]),
                "job": copy.deepcopy(cell.get("job")),
                "current_adapt": {
                    "execution_id": str(current_adapt["execution_id"]),
                    "exact_same_cutoff_energy": float(exact),
                    "points": normalized_adapt,
                    "marker": {
                        "k": TARGET_HORIZON,
                        "error": float(marker_error),
                        "policy": "terminal_common_horizon",
                    },
                    "status": "complete / reused from Page 12",
                    "full_source_horizon": int(normalized_adapt_full[-1]["k"]),
                    "plotted_horizon": TARGET_HORIZON,
                    "full_source_point_count": len(normalized_adapt_full),
                    "plotted_point_count": len(normalized_adapt),
                    "display_crop": "common_comparator_horizon_k_le_50",
                    "source": copy.deepcopy(current_adapt["source"]),
                    "source_limitations": copy.deepcopy(source_limitations),
                },
                "status": str(route["status"]),
            }
        )

        def cell_status(policy: str) -> str:
            result = completed.get(regime, {}).get(policy)
            if result is None:
                return "pending / awaiting authenticated receipt"
            return (
                f"complete / authenticated k={int(result['terminal']['k'])}, "
                f"|dE|={float(result['terminal']['error']):.2e}"
            )

        matrix.append(
            {
                "regime_id": regime,
                "regime_label": REGIME_LABELS[regime],
                "nph": int(cell["nph"]),
                "current_adapt": "complete / reused from Page 12",
                "plateau_insertion": "complete / reused from Page 12",
                "reference": "complete / reused from Page 12",
                "always_insertion": cell_status(EXPECTED_POLICIES[0]),
                "append_always": cell_status(EXPECTED_POLICIES[1]),
            }
        )

    always_count = sum(
        EXPECTED_POLICIES[0] in completed.get(regime, {}) for regime in REGIME_ORDER
    )
    append_count = sum(
        EXPECTED_POLICIES[1] in completed.get(regime, {}) for regime in REGIME_ORDER
    )
    authenticated_count = always_count + append_count
    receipt_revision = _canonical_sha256(
        {
            result["run_id"]: result["receipt_sha256"]
            for result in sorted(results, key=lambda row: int(row["proc_id"]))
        }
    )
    unsigned: dict[str, Any] = {
        "schema": ADAPTER_SCHEMA,
        "status": f"provisional_page12_{authenticated_count}_of_12_authenticated",
        "page_ids": [PAGE18_ID],
        "run_class": "diagnostic",
        "paper_evidence_adopted": False,
        "plateau_reference_reused_not_rerun": True,
        "parameter_manifest": {
            "model": "Hubbard--Holstein L=2",
            "boundary": "open",
            "boson_encoding": "binary",
            "optimizer": "Powell",
            "representation": "Page 12 global-singleton Phase-0 route",
            "error_metric": "same-cutoff absolute energy error",
            "reference_curves": [
                "current Append-ADAPT baseline",
                "current plateau-insertion RA-ADAPT",
            ],
            "comparator_policies": list(EXPECTED_POLICIES),
            "comparator_horizon": TARGET_HORIZON,
            "display_crop": "all_curves_common_horizon_k_le_50",
        },
        "campaign_counts": {
            "planned_comparator_cells": 12,
            "authenticated_comparator_cells": authenticated_count,
            "always_insertion_authenticated": always_count,
            "append_always_authenticated": append_count,
            "pending_comparator_cells": 12 - authenticated_count,
        },
        "receipt_evidence_revision": receipt_revision,
        "matrix": matrix,
        "completed_comparators": completed,
        "reference_cells": references,
        "sources": {
            "page12_reference_adapter": {
                **(
                    binding(REFERENCE_ADAPTER)
                    if REFERENCE_ADAPTER.is_file()
                    else {"path": str(REFERENCE_ADAPTER)}
                ),
                "canonical_sha256": reference_adapter["sha256"],
            },
            "fixed_comparator_package": {
                "path": str(PACKAGE_MANIFEST.resolve()),
                "canonical_sha256": (
                    _load_package()[0]["sha256"]
                    if PACKAGE_MANIFEST.is_file()
                    else None
                ),
            },
            "closure_receipts": {
                result["run_id"]: copy.deepcopy(result["source"])
                for result in results
            },
        },
        "limitations": [
            (
                f"{authenticated_count}/12 comparator cells have authenticated "
                "finalizer receipts and are plotted."
            ),
            "All absent comparator cells remain visibly pending in every panel/table.",
            (
                "The current Append-ADAPT baseline and Page-12 plateau-insertion "
                "reference are reused and were not rerun."
            ),
            *reference_limitations,
            "No paper-evidence adoption or insertion-policy conclusion is implied.",
        ],
    }
    return {**unsigned, "sha256": _canonical_sha256(unsigned)}


def _style_table(table: Any) -> None:
    table.auto_set_font_size(False)
    for (row, _column), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#E8E8E8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")


def render_page(adapter: Mapping[str, Any]) -> None:
    """Render one established-format dense six-regime Page 18."""

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    _atomic_json(ADAPTER_PATH, adapter)
    completed = adapter["completed_comparators"]
    matrix = {row["regime_id"]: row for row in adapter["matrix"]}
    styles = {
        EXPECTED_POLICIES[0]: {"color": ORANGE, "marker": "D"},
        EXPECTED_POLICIES[1]: {"color": MAGENTA, "marker": "o"},
    }

    def annotation(regime: str, policy: str) -> str:
        short = "insert-always" if policy == EXPECTED_POLICIES[0] else "append-always"
        result = completed.get(regime, {}).get(policy)
        if result is None:
            return f"{short}: PENDING authenticated receipt"
        return (
            f"{short}: COMPLETE k={int(result['terminal']['k'])}, "
            f"|dE|={float(result['terminal']['error']):.2e}"
        )

    mpl.rcParams.update({"font.family": "serif", "font.size": 7.2})
    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        3,
        3,
        left=0.065,
        right=0.96,
        top=0.84,
        bottom=0.10,
        height_ratios=(1.0, 1.0, 0.56),
        hspace=0.40,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    for index, (axis, reference) in enumerate(
        zip(axes, adapter["reference_cells"], strict=True)
    ):
        regime = str(reference["regime_id"])
        current_adapt = reference["current_adapt"]
        adapt_points = current_adapt["points"]
        axis.plot(
            [point["k"] for point in adapt_points],
            [max(float(point["error"]), PLOT_FLOOR) for point in adapt_points],
            color=BLUE,
            lw=1.35,
            linestyle="-",
        )
        adapt_marker = current_adapt["marker"]
        axis.scatter(
            [adapt_marker["k"]],
            [max(float(adapt_marker["error"]), PLOT_FLOOR)],
            color=BLUE,
            marker="o",
            s=20,
            zorder=5,
        )
        ref_points = reference["points"]
        axis.plot(
            [point["k"] for point in ref_points],
            [max(float(point["error"]), PLOT_FLOOR) for point in ref_points],
            color=GREEN,
            lw=1.55,
            linestyle="-",
        )
        ref_marker = reference["marker"]
        axis.scatter(
            [ref_marker["k"]],
            [max(float(ref_marker["error"]), PLOT_FLOOR)],
            color=GREEN,
            marker="s",
            s=20,
            zorder=5,
        )
        for policy in EXPECTED_POLICIES:
            result = completed.get(regime, {}).get(policy)
            if result is None:
                continue
            style = styles[policy]
            points = result["points"]
            axis.plot(
                [point["k"] for point in points],
                [max(float(point["error"]), PLOT_FLOOR) for point in points],
                color=style["color"],
                lw=1.75,
                linestyle="-",
            )
            marker = result["marker"]
            axis.scatter(
                [marker["k"]],
                [max(float(marker["error"]), PLOT_FLOOR)],
                color=style["color"],
                marker=style["marker"],
                s=27,
                zorder=6,
            )
        axis.text(
            0.97,
            0.07,
            "\n".join(annotation(regime, policy) for policy in EXPECTED_POLICIES),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.55,
            color=GRAY,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )
        axis.set_yscale("log")
        axis.set_xlim(0, TARGET_HORIZON)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.grid(True, which="major", alpha=0.22, lw=0.5)
        axis.set_title(
            f"{reference['regime_label']} ($n_{{ph}}={reference['nph']}$)",
            fontsize=8.1,
        )
        if index // 3 == 1:
            axis.set_xlabel(r"ADAPT iteration $k$")
        if index % 3 == 0:
            axis.set_ylabel(r"same-cutoff $|\Delta E(k)|$")

    counts = adapter["campaign_counts"]
    fig.suptitle(
        (
            "Page 12 global-singleton comparators: "
            r"same-cutoff $|\Delta E(k)|$ vs current ADAPT"
        ),
        fontsize=11.2,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.948,
        (
            f"Authenticated curves: {counts['authenticated_comparator_cells']}/12 "
            f"(insertion always {counts['always_insertion_authenticated']}; "
            f"append always {counts['append_always_authenticated']}); "
            f"pending {counts['pending_comparator_cells']}"
        ),
        ha="center",
        color=RED,
        fontsize=7.2,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE, lw=1.35, marker="o",
                label="current ADAPT (Append-ADAPT baseline)",
            ),
            Line2D(
                [0], [0], color=GREEN, lw=1.55, marker="s",
                label="current plateau-insertion RA-ADAPT",
            ),
            Line2D(
                [0], [0], color=ORANGE, lw=1.75, marker="D",
                label="RA-ADAPT insertion always",
            ),
            Line2D(
                [0], [0], color=MAGENTA, lw=1.75, marker="o",
                label="RA-ADAPT append-only insertion (append always)",
            ),
        ],
        title=(
            "Marker = first effective plateau prefix when defined; otherwise terminal plotted point"
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.920),
        ncol=4,
        frameon=False,
        fontsize=6.7,
        title_fontsize=5.8,
    )

    table_axis = fig.add_subplot(grid[2, :])
    table_axis.axis("off")
    rows = []
    for reference in adapter["reference_cells"]:
        regime = str(reference["regime_id"])
        row = matrix[regime]
        rows.append(
            [
                reference["regime_label"],
                (
                    f"k={int(reference['current_adapt']['marker']['k'])}; "
                    f"{float(reference['current_adapt']['marker']['error']):.2e}"
                ),
                f"k={int(reference['marker']['k'])}; {float(reference['marker']['error']):.2e}",
                row["always_insertion"].replace("authenticated ", "auth. "),
                row["append_always"].replace("authenticated ", "auth. "),
            ]
        )
    table = table_axis.table(
        cellText=rows,
        colLabels=[
            "Regime",
            r"Current ADAPT $(k,|\Delta E|)$",
            r"Plateau insertion $(k,|\Delta E|)$",
            "RA insertion always",
            "RA append-only (append always)",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.14, 0.18, 0.18, 0.25, 0.25),
    )
    _style_table(table)
    table.set_fontsize(5.8)
    table.scale(1.0, 0.92)
    for row_index, reference in enumerate(adapter["reference_cells"], 1):
        regime = str(reference["regime_id"])
        for column, policy, color in (
            (3, EXPECTED_POLICIES[0], "#FFF0D9"),
            (4, EXPECTED_POLICIES[1], "#FCEAF5"),
        ):
            if completed.get(regime, {}).get(policy) is not None:
                table[(row_index, column)].set_facecolor(color)
                table[(row_index, column)].set_text_props(weight="bold")

    reference_sha = adapter["sources"]["page12_reference_adapter"][
        "canonical_sha256"
    ]
    fig.text(
        0.5,
        0.050,
        (
            "Page 12 global singleton; HH L=2; open boundary; binary bosons; Powell; "
            f"reference {reference_sha[:12]}...; receipt revision "
            f"{adapter['receipt_evidence_revision'][:12]}...; same-cutoff exact reference. "
            "Current Append-ADAPT and plateau-insertion curves are reused from Page 12. "
            "Comparator curves require self-digested finalizer receipts plus full archive "
            "member hash/size closure."
        ),
        ha="center",
        fontsize=5.6,
        color=GRAY,
    )
    fig.text(
        0.5,
        0.024,
        (
            "PROVISIONAL DIAGNOSTIC - every absent cell remains PENDING; the Page-12 "
            "reference curves were reused, not rerun; no paper-evidence adoption or policy "
            "conclusion is implied."
        ),
        ha="center",
        fontsize=6.1,
        color=RED,
        fontweight="bold",
    )

    # Reuse the established Page-17 output mechanism without importing its
    # campaign contract or pinned execution adapters.
    from pipelines.reporting import append_paper_i_completed_beam_noise_pages

    append_paper_i_completed_beam_noise_pages._save_page(
        fig, png_path=PAGE18_PNG, pdf_path=PAGE18_PDF
    )
    plt.close(fig)


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    payload = b"" if contents is None else contents.get_data()
    return hashlib.sha256(payload).hexdigest()


def _with_report_mutation_lock(function: Any) -> Any:
    def locked(*args: Any, **kwargs: Any) -> Any:
        REPORT_MUTATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REPORT_MUTATION_LOCK_PATH.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            return function(*args, **kwargs)

    return locked


@_with_report_mutation_lock
def append_or_replace_page(adapter: Mapping[str, Any]) -> dict[str, Any]:
    """Append Page 18 once, then replace it without disturbing later pages."""

    from pypdf import PdfReader, PdfWriter

    provenance = load(TARGET_PROVENANCE)
    current = binding(TARGET_PDF)
    layout = provenance.get("layout")
    outputs = provenance.get("outputs")
    declared = outputs.get("partial_progress_pdf") if isinstance(outputs, Mapping) else None
    page_count = int(layout.get("page_count", -1)) if isinstance(layout, Mapping) else -1
    if (
        not isinstance(layout, Mapping)
        or not isinstance(declared, Mapping)
        or current["sha256"] != declared.get("sha256")
        or current["size_bytes"] != declared.get("size_bytes")
        or layout.get("page_17") != PAGE17_ID
        or not (
            page_count == 17
            or (page_count == 18 and layout.get("page_18") == PAGE18_ID)
            or (
                page_count == 19
                and layout.get("page_18") == PAGE18_ID
                and layout.get("page_19") == PAGE19_ID
            )
        )
    ):
        raise WatchError("target PDF/provenance is not the current dense Page-17 state")
    original = PdfReader(str(TARGET_PDF), strict=False)
    page18 = PdfReader(str(PAGE18_PDF), strict=False)
    if len(original.pages) != page_count or len(page18.pages) != 1:
        raise WatchError("Page-18 update requires one one-page snapshot asset")
    preserved_hashes = [_page_content_sha256(page) for page in original.pages[:17]]
    writer = PdfWriter()
    for page in original.pages[:17]:
        writer.add_page(page)
    writer.add_page(page18.pages[0])
    if page_count == 19:
        writer.add_page(original.pages[18])

    updated_page_count = 19 if page_count == 19 else 18
    preserved_page19_hash = (
        _page_content_sha256(original.pages[18]) if page_count == 19 else None
    )

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
        if len(combined.pages) != updated_page_count:
            raise WatchError(
                f"combined report must contain exactly {updated_page_count} pages"
            )
        if [_page_content_sha256(page) for page in combined.pages[:17]] != preserved_hashes:
            raise WatchError("Page-18 update changed Pages 1--17")
        if (
            page_count == 19
            and _page_content_sha256(combined.pages[18]) != preserved_page19_hash
        ):
            raise WatchError("Page-18 update changed Page 19")

        updated = copy.deepcopy(provenance)
        updated["layout"]["page_17"] = PAGE17_ID
        updated["layout"]["page_18"] = PAGE18_ID
        updated["layout"]["page_count"] = updated_page_count
        updated["phase0_page12_insertion_comparator_snapshot"] = {
            "schema": "paper_i_ra_adapt_page12_insertion_comparator_progress_report_v1",
            "status": adapter["status"],
            "page_ids": [PAGE18_ID],
            "run_class": "diagnostic",
            "paper_evidence_adopted": False,
            "adapter": {**binding(ADAPTER_PATH), "canonical_sha256": adapter["sha256"]},
            "campaign_counts": copy.deepcopy(adapter["campaign_counts"]),
            "receipt_evidence_revision": adapter.get("receipt_evidence_revision"),
            "matrix": copy.deepcopy(adapter["matrix"]),
            "completed_comparators": copy.deepcopy(adapter["completed_comparators"]),
            "sources": copy.deepcopy(adapter["sources"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "outputs": {
                "page18_pdf": binding(PAGE18_PDF),
                "page18_png": binding(PAGE18_PNG),
            },
        }
        updated["outputs"]["page12_insertion_comparator_snapshot_adapter"] = {
            **binding(ADAPTER_PATH),
            "canonical_sha256": adapter["sha256"],
        }
        updated["outputs"]["page12_insertion_comparator_snapshot_page18_pdf"] = binding(
            PAGE18_PDF
        )
        updated["outputs"]["page12_insertion_comparator_snapshot_page18_png"] = binding(
            PAGE18_PNG
        )
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

        # Recheck the optimistic source binding immediately before the paired
        # replacement.  This catches a completed concurrent Page-17 refresh.
        if binding(TARGET_PDF) != current:
            raise WatchError("target report changed during Page-18 preparation")
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
        "status": "updated_existing_report_in_place",
        "page_count": updated_page_count,
        "preserved_page_count": updated_page_count - 1,
        "authenticated_comparator_count": adapter["campaign_counts"].get(
            "authenticated_comparator_cells"
        ),
        "pdf": binding(TARGET_PDF),
        "provenance": binding(TARGET_PROVENANCE),
    }


def refresh_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    adapter = build_adapter(results)
    render_page(adapter)
    result = append_or_replace_page(adapter)
    return {**result, "receipt_evidence_revision": adapter["receipt_evidence_revision"]}


def _reported_revision() -> tuple[set[str], str | None]:
    if not TARGET_PROVENANCE.is_file() or TARGET_PROVENANCE.is_symlink():
        return set(), None
    provenance = load(TARGET_PROVENANCE)
    layout = provenance.get("layout")
    if (
        not isinstance(layout, Mapping)
        or layout.get("page_count") not in {18, 19}
        or layout.get("page_18") != PAGE18_ID
        or (
            layout.get("page_count") == 19
            and layout.get("page_19") != PAGE19_ID
        )
    ):
        return set(), None
    report = provenance.get("phase0_page12_insertion_comparator_snapshot")
    if not isinstance(report, Mapping):
        return set(), None
    completed = report.get("completed_comparators")
    run_ids: set[str] = set()
    if isinstance(completed, Mapping):
        for policies in completed.values():
            if not isinstance(policies, Mapping):
                continue
            for row in policies.values():
                if isinstance(row, Mapping) and isinstance(row.get("run_id"), str):
                    run_ids.add(str(row["run_id"]))
    revision = report.get("receipt_evidence_revision")
    if revision is not None and not _is_sha256(revision):
        raise WatchError("reported Page-12 evidence revision is malformed")
    return run_ids, revision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {**copy.deepcopy(dict(value)), "updated_at_utc": _utc_now()}
    payload = {**unsigned, "sha256": _canonical_sha256(unsigned)}
    _atomic_json(WATCH_STATUS_PATH, payload)
    artifact_root = os.environ.get("REMOTE_ARTIFACT_DIR")
    if artifact_root:
        _atomic_json(
            Path(artifact_root) / "page12_page18_auto_refresh_status.json", payload
        )
    return payload


def _load_previous_status() -> dict[str, Any] | None:
    if not WATCH_STATUS_PATH.exists() and not WATCH_STATUS_PATH.is_symlink():
        return None
    if not WATCH_STATUS_PATH.is_file() or WATCH_STATUS_PATH.is_symlink():
        raise WatchError("Page-12 watcher status is unsafe")
    value = load(WATCH_STATUS_PATH)
    _verify_self_digest(value, label="Page-12 watcher status")
    if value.get("schema") != STATUS_SCHEMA:
        raise WatchError("Page-12 watcher status schema drifted")
    return value


def _next_poll(
    previous: Mapping[str, Any] | None,
    *,
    fingerprint: str,
    base: float,
    maximum: float,
) -> float:
    if previous is None or previous.get("source_state_fingerprint") != fingerprint:
        return base
    prior = previous.get("next_poll_seconds")
    if not isinstance(prior, (int, float)):
        return base
    return min(maximum, max(base, float(prior) * 1.6))


def watch(*, receipt_dir: Path, poll_seconds: float, once: bool) -> int:
    previous = _load_previous_status()
    while True:
        try:
            results = authenticated_inventory(receipt_dir)
            run_ids = {str(result["run_id"]) for result in results}
            revision = _canonical_sha256(
                {
                    result["run_id"]: result["receipt_sha256"]
                    for result in results
                }
            )
            reported_ids, reported_revision = _reported_revision()
            refresh_result = None
            if run_ids != reported_ids or revision != reported_revision:
                refresh_result = refresh_report(results)
                reported_ids, reported_revision = _reported_revision()
                if reported_ids != run_ids or reported_revision != revision:
                    raise WatchError("Page-18 updater did not publish authenticated revision")
            if not results:
                status = "waiting_for_first_authenticated_receipt"
            elif len(results) == 12:
                status = "passed_all_twelve_receipts_refreshed"
            else:
                status = "watching_for_next_authenticated_receipt"
            fingerprint = _canonical_sha256(
                {result["run_id"]: result["receipt_sha256"] for result in results}
            )
            next_poll = None
            if status != "passed_all_twelve_receipts_refreshed":
                next_poll = _next_poll(
                    previous,
                    fingerprint=fingerprint,
                    base=poll_seconds,
                    maximum=DEFAULT_MAX_POLL_SECONDS,
                )
            payload = {
                "schema": STATUS_SCHEMA,
                "status": status,
                "credentials_used": False,
                "scheduler_or_scientific_action_performed": False,
                "receipt_dir": str(receipt_dir.resolve()),
                "authenticated_receipt_count": len(results),
                "authenticated_run_ids": sorted(run_ids),
                "reported_run_ids": sorted(reported_ids),
                "receipt_evidence_revision": revision,
                "reported_evidence_revision": reported_revision,
                "source_state_fingerprint": fingerprint,
                "next_poll_seconds": next_poll,
                "last_refresh_result": refresh_result,
                "last_error": None,
            }
            previous = _write_status(payload)
            if status == "passed_all_twelve_receipts_refreshed" or once:
                return 0
            assert next_poll is not None
            time.sleep(next_poll)
        except (OSError, WatchError, json.JSONDecodeError) as exc:
            _write_status(
                {
                    "schema": STATUS_SCHEMA,
                    "status": "watcher_authentication_failed",
                    "credentials_used": False,
                    "scheduler_or_scientific_action_performed": False,
                    "receipt_dir": str(receipt_dir.resolve()),
                    "authenticated_receipt_count": 0,
                    "authenticated_run_ids": [],
                    "reported_run_ids": [],
                    "receipt_evidence_revision": None,
                    "reported_evidence_revision": None,
                    "source_state_fingerprint": None,
                    "next_poll_seconds": None,
                    "last_refresh_result": None,
                    "last_error": str(exc),
                }
            )
            print(str(exc), file=os.sys.stderr, flush=True)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        epilog=f"Repository root: {REPO_ROOT}",
    )
    parser.add_argument("--receipt-dir", type=Path, default=RECEIPT_DIR)
    parser.add_argument("--poll-seconds", type=float, default=MIN_POLL_SECONDS)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < MIN_POLL_SECONDS:
        raise SystemExit("--poll-seconds must be at least 30")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Page-12 Page-18 watcher is already running.", file=os.sys.stderr)
            return 2
        return watch(
            receipt_dir=args.receipt_dir.resolve(),
            poll_seconds=args.poll_seconds,
            once=args.once,
        )


if __name__ == "__main__":
    raise SystemExit(main())
