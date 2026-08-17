#!/usr/bin/env python3
"""Locally authenticate and seal one fixed Page-12 comparator archive.

The helper is deliberately scheduler- and network-inert.  Its only mutating
mode atomically publishes a local, self-digested closure receipt after all
sealed package, activation, transfer-identity, archive, and worker contracts
have passed.  The proc number selects one of the twelve immutable queue rows;
callers cannot supply alternate package, activation, archive, or receipt paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, BinaryIO, Mapping
import uuid


HELPER_PATH = Path(__file__).resolve()
SOURCE_REPO_ROOT = HELPER_PATH.parents[2]
REPAIR_RELATIVE = Path("chtc/paper_i_ra_adapt_repair_20260727")
PACKAGE_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
ACTIVATION_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_"
    "20260812_v1_chtc_activation_v1"
)
ARCHIVE_DIR_RELATIVE = REPAIR_RELATIVE / (
    "retrieved_page12_insertion_comparators_20260813"
)
EVIDENCE_DIR_RELATIVE = REPAIR_RELATIVE / (
    "page12_insertion_comparator_closure_evidence"
)
RECEIPT_DIR_RELATIVE = REPAIR_RELATIVE / (
    "page12_insertion_comparator_closure_receipts"
)
SUBMISSION_RECEIPT_RELATIVE = REPAIR_RELATIVE / (
    "paper_i_ra_adapt_insertion_comparators_all24_"
    "20260812_submission_receipt_9644571_9647385_9647386.json"
)

PACKAGE_DIR = SOURCE_REPO_ROOT / PACKAGE_RELATIVE
ACTIVATION_DIR = SOURCE_REPO_ROOT / ACTIVATION_RELATIVE
SUBMISSION_RECEIPT_PATH = SOURCE_REPO_ROOT / SUBMISSION_RECEIPT_RELATIVE

CLUSTER_ID = 9_647_385
ROW_COUNT = 12
TARGET_HORIZON = 50
EXPECTED_PACKAGE_FILE_SHA256 = (
    "4cf3df426f6b3545e51b90c0ffaf1f0755b989ae3644a067ca8cbcf98b5026bd"
)
EXPECTED_PACKAGE_CANONICAL_SHA256 = (
    "efce225efdc04653e8fca7e34eb3f467d4a6ec594e2130cde4bbea45e3d040e9"
)
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
)
EXPECTED_QUEUE_SHA256 = (
    "406610e80d3f73521225da7852f1ee57414ee2ee1cfd0d4c73979d4b9f47527c"
)
EXPECTED_ACTIVATION_FILE_SHA256 = (
    "104a22a774cee5526c9272082ea884745e4f851c55097396ea1de2eeeaf47ca4"
)
EXPECTED_ACTIVATION_CANONICAL_SHA256 = (
    "9aa36c3362257dfdcd8624bf091adfbaae28edb06e0abadcb8d6b6936533a36d"
)
EXPECTED_SUBMISSION_RECEIPT_SHA256 = (
    "777ca1045664c6f843704b1f55b269a77932d5bd89483ca99b65daa5faf12336"
)
EXPECTED_PACKAGE_ID = (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
EXPECTED_CAMPAIGN_ID = (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1"
)
EXPECTED_BUNDLE_ID = "ra_adapt_page12_insertion_comparators_r50_20260812_v1"
EXPECTED_POLICIES = (
    "always_commutation_reduced",
    "append_only",
)
EXPECTED_REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
WORKER_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_worker_receipt_v1"
)
MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_execution_manifest_v1"
)
ACTIVATION_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_activation_manifest_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_"
    "execution_authorization_v1"
)
CLOSURE_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_page12_insertion_comparator_closure_receipt_v1"
)
CLOSURE_RECEIPT_STATUS = (
    "passed_authenticated_page12_insertion_comparator_closure"
)
REMOTE_OUTPUT_ROOT = (
    "osdf:///chtc/staging/j/jsstrobel/"
    "paper_i_ra_adapt_page12_insertion_comparators_20260812_v1/outputs/"
    "transfer"
)
JSON_CAPTURE_LIMIT_BYTES = 4 * 1024 * 1024


class ClosureError(RuntimeError):
    """Raised when Page-12 closure evidence fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    return {
        **unsigned,
        "sha256": hashlib.sha256(_canonical_bytes(unsigned)).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ClosureError(f"{label} is absent or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object: {path}")
    return value


def _verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()
    if value.get("sha256") != actual:
        raise ClosureError(f"{label} self digest drifted")
    return actual


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClosureError(f"{label} path is absent")
    raw = value
    while raw.startswith("./"):
        raw = raw[2:]
    pure = PurePosixPath(raw)
    if (
        not raw
        or raw == "."
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in raw
    ):
        raise ClosureError(f"{label} path is unsafe: {value!r}")
    return pure.as_posix()


def _verify_binding(
    root: Path,
    row: Any,
    *,
    label: str,
    canonical: bool,
) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(row, Mapping):
        raise ClosureError(f"{label} binding is absent")
    relative = _safe_relative(row.get("path"), label=label)
    path = root / Path(*PurePosixPath(relative).parts)
    if not path.is_file() or path.is_symlink():
        raise ClosureError(f"{label} binding target is absent or unsafe: {path}")
    if (
        type(row.get("size_bytes")) is not int
        or path.stat().st_size != row["size_bytes"]
        or not isinstance(row.get("sha256"), str)
        or _sha256_file(path) != row["sha256"]
    ):
        raise ClosureError(f"{label} file binding drifted")
    value: dict[str, Any] | None = None
    if canonical:
        value = _load_object(path, label=label)
        digest = _verify_self_digest(value, label=label)
        if row.get("canonical_sha256") != digest:
            raise ClosureError(f"{label} canonical binding drifted")
    return path, value


def _queue_rows() -> list[list[str]]:
    path = PACKAGE_DIR / "queue.tsv"
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != EXPECTED_QUEUE_SHA256
    ):
        raise ClosureError("sealed Page-12 queue drifted")
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != ROW_COUNT or any(len(row) != 8 for row in rows):
        raise ClosureError("sealed Page-12 queue row shape drifted")
    expected_order = [
        (policy, regime, nph)
        for policy in EXPECTED_POLICIES
        for regime, nph in EXPECTED_REGIMES
    ]
    for row, (policy, regime, nph) in zip(rows, expected_order, strict=True):
        execution_id, job_path, protocol_path, job_file_sha, *_resources = row
        if (
            f"__{regime}__nph{nph}__" not in execution_id
            or not execution_id.endswith(f"_{policy}")
            or job_path != f"jobs/{execution_id}.json"
            or not protocol_path.endswith(f"/{execution_id}.json")
            or len(job_file_sha) != 64
        ):
            raise ClosureError("sealed Page-12 proc mapping drifted")
    return rows


def _load_package_and_target(proc: int) -> dict[str, Any]:
    package_path = PACKAGE_DIR / "package_manifest.json"
    if (
        not package_path.is_file()
        or package_path.is_symlink()
        or _sha256_file(package_path) != EXPECTED_PACKAGE_FILE_SHA256
    ):
        raise ClosureError("fixed Page-12 package manifest file drifted")
    package = _load_object(package_path, label="fixed Page-12 package manifest")
    _verify_self_digest(package, label="fixed Page-12 package manifest")
    if (
        package.get("sha256") != EXPECTED_PACKAGE_CANONICAL_SHA256
        or package.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_package_manifest_v1"
        or package.get("status") != "passed_inert_twelve_cells"
        or package.get("package_id") != EXPECTED_PACKAGE_ID
        or package.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or package.get("bundle_id") != EXPECTED_BUNDLE_ID
        or package.get("comparator_policies") != list(EXPECTED_POLICIES)
        or package.get("row_count") != ROW_COUNT
        or package.get("execution_authorized") is not False
        or package.get("submission_authorized") is not False
        or package.get("submitted") is not False
        or package.get("source_archive", {}).get("sha256")
        != EXPECTED_SOURCE_ARCHIVE_SHA256
        or package.get("queue", {}).get("sha256") != EXPECTED_QUEUE_SHA256
    ):
        raise ClosureError("fixed Page-12 package identity drifted")

    for key, label in (
        ("execution_plan", "execution plan"),
        ("source_archive_manifest", "source archive manifest"),
        ("source_lock_audit", "source-lock audit"),
        ("non_insertion_equality_audit", "non-insertion equality audit"),
        ("bundle_manifest", "bundle manifest"),
        ("bundle_source_locks", "bundle source locks"),
        ("bundle_expected_artifacts", "bundle expected artifacts"),
        ("bundle_validation_report", "bundle validation report"),
    ):
        _verify_binding(PACKAGE_DIR, package.get(key), label=label, canonical=True)
    _verify_binding(
        PACKAGE_DIR,
        package.get("source_archive"),
        label="source archive",
        canonical=False,
    )

    rows = _queue_rows()
    execution_ids = [row[0] for row in rows]
    if package.get("execution_ids") != execution_ids:
        raise ClosureError("package execution/proc order drifted")
    job_bindings = package.get("jobs")
    protocol_bindings = package.get("protocols")
    if (
        not isinstance(job_bindings, list)
        or not isinstance(protocol_bindings, list)
        or len(job_bindings) != ROW_COUNT
        or len(protocol_bindings) != ROW_COUNT
    ):
        raise ClosureError("package job/protocol inventory drifted")
    jobs_by_id = {
        row.get("execution_id"): row
        for row in job_bindings
        if isinstance(row, Mapping)
    }
    protocols_by_id = {
        row.get("execution_id"): row
        for row in protocol_bindings
        if isinstance(row, Mapping)
    }
    if set(jobs_by_id) != set(execution_ids) or set(protocols_by_id) != set(execution_ids):
        raise ClosureError("package job/protocol execution inventory drifted")

    run_id = execution_ids[proc]
    job_path, job_value = _verify_binding(
        PACKAGE_DIR,
        jobs_by_id[run_id],
        label="fixed proc job",
        canonical=True,
    )
    assert job_value is not None
    job = job_value
    queue = rows[proc]
    if (
        job_path.relative_to(PACKAGE_DIR).as_posix() != queue[1]
        or _sha256_file(job_path) != queue[3]
        or job.get("schema")
        != "paper_i_ra_adapt_gradient_phase0_phase23_qiskit_job_v1"
        or job.get("package_id") != EXPECTED_PACKAGE_ID
        or job.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or job.get("bundle_id") != EXPECTED_BUNDLE_ID
        or job.get("execution_id") != run_id
        or job.get("cell_id") != run_id
        or job.get("target_horizon") != TARGET_HORIZON
        or job.get("fresh_start_contract")
        != {"kind": "fresh_start", "resume_archive": None, "source_checkpoint": None}
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
        or job.get("submitted") is not False
    ):
        raise ClosureError("fixed proc job identity drifted")

    protocol_path, protocol_value = _verify_binding(
        PACKAGE_DIR,
        protocols_by_id[run_id],
        label="fixed proc protocol",
        canonical=True,
    )
    assert protocol_value is not None
    protocol = protocol_value
    route = protocol.get("route_contract")
    if not isinstance(route, Mapping):
        raise ClosureError("fixed proc route contract is absent")
    _verify_self_digest(route, label="fixed proc route contract")
    insertion = job.get("comparator_policy")
    expected_mode = (
        "full_commutation_reduced"
        if insertion == "always_commutation_reduced"
        else "append_only"
    )
    request = protocol.get("request")
    execution = route.get("execution_settings")
    semantic = route.get("semantic_invariants")
    lineage = route.get("lineage_authority")
    if (
        protocol_path.relative_to(PACKAGE_DIR).as_posix() != queue[2]
        or protocol.get("schema") != "paper_i_ra_adapt_resolved_protocol_v1"
        or protocol.get("sha256") != job.get("protocol_sha256")
        or _sha256_file(protocol_path) != job.get("protocol_file_sha256")
        or route.get("sha256") != job.get("route_contract_sha256")
        or package.get("route_contract_sha256_by_execution_id", {}).get(run_id)
        != route.get("sha256")
        or job.get("typed_insertion_kind") != insertion
        or job.get("runtime_insertion_mode") != expected_mode
        or not isinstance(request, Mapping)
        or request.get("method", {}).get("insertion", {}).get("kind") != insertion
        or not isinstance(execution, Mapping)
        or execution.get("adapt_insertion_mode") != expected_mode
        or not isinstance(semantic, Mapping)
        or not isinstance(lineage, Mapping)
        or f"typed_insertion_comparator:{insertion}"
        not in lineage.get("only_intended_scientific_changes", [])
    ):
        raise ClosureError("fixed proc protocol/route/insertion identity drifted")
    if insertion == "always_commutation_reduced" and (
        semantic.get("insertion_position_scope")
        != "full_logical_ansatz_commutation_classes_every_depth_v2"
        or semantic.get("insertion_equivalence_policy")
        != "termwise_cross_component_commutation_earliest_representative_v1"
    ):
        raise ClosureError("always-insertion reduction identity drifted")
    if insertion == "append_only" and any(
        key in semantic
        for key in ("insertion_position_scope", "insertion_equivalence_policy")
    ):
        raise ClosureError("append-only insertion identity drifted")
    return {
        "package": package,
        "package_path": package_path,
        "job": job,
        "job_path": job_path,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "run_id": run_id,
    }


def _load_activation(target: Mapping[str, Any]) -> dict[str, Any]:
    path = ACTIVATION_DIR / "activation_manifest.json"
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != EXPECTED_ACTIVATION_FILE_SHA256
    ):
        raise ClosureError(f"exact Page-12 activation snapshot is absent or drifted: {path}")
    activation = _load_object(path, label="exact Page-12 activation manifest")
    _verify_self_digest(activation, label="exact Page-12 activation manifest")
    if (
        activation.get("sha256") != EXPECTED_ACTIVATION_CANONICAL_SHA256
        or activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("status") != "passed_activation_prepared_no_submission"
        or activation.get("package_id") != EXPECTED_PACKAGE_ID
        or activation.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or activation.get("bundle_id") != EXPECTED_BUNDLE_ID
        or activation.get("package_manifest_sha256")
        != EXPECTED_PACKAGE_CANONICAL_SHA256
        or activation.get("authorization_count") != ROW_COUNT
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or activation.get("paper_evidence_adoption_authorized") is not False
        or activation.get("submitted") is not False
    ):
        raise ClosureError("exact Page-12 activation identity drifted")
    request_path, request = _verify_binding(
        ACTIVATION_DIR,
        activation.get("activation_request"),
        label="activation request",
        canonical=True,
    )
    _probe_path, probe = _verify_binding(
        ACTIVATION_DIR,
        activation.get("image_runtime_probe"),
        label="activation image-runtime probe",
        canonical=True,
    )
    assert request is not None and probe is not None
    expected_ids = _queue_rows()
    if (
        request.get("requested_execution_ids") != [row[0] for row in expected_ids]
        or request.get("execution_authorized") is not True
        or request.get("submission_authorized") is not True
        or request.get("paper_evidence_adoption_authorized") is not False
        or probe.get("deep_worker_preflight_count") != ROW_COUNT
        or probe.get("deep_worker_preflight_runtime") != "pinned_execution_image"
        or probe.get("launch_ready") is not True
    ):
        raise ClosureError("activation request/probe identity drifted")
    raw_authorizations = activation.get("authorizations")
    if not isinstance(raw_authorizations, list) or len(raw_authorizations) != ROW_COUNT:
        raise ClosureError("activation authorization inventory drifted")
    by_id: dict[str, dict[str, Any]] = {}
    for row in raw_authorizations:
        if not isinstance(row, Mapping) or not isinstance(row.get("execution_id"), str):
            raise ClosureError("activation authorization binding drifted")
        execution_id = row["execution_id"]
        _auth_path, authority = _verify_binding(
            ACTIVATION_DIR,
            row,
            label=f"authorization {execution_id}",
            canonical=True,
        )
        assert authority is not None
        if (
            authority.get("schema") != AUTHORIZATION_SCHEMA
            or authority.get("execution_id") != execution_id
            or authority.get("package_id") != EXPECTED_PACKAGE_ID
            or authority.get("campaign_id") != EXPECTED_CAMPAIGN_ID
            or authority.get("bundle_id") != EXPECTED_BUNDLE_ID
            or authority.get("package_manifest_sha256")
            != EXPECTED_PACKAGE_CANONICAL_SHA256
            or authority.get("source_archive_sha256")
            != EXPECTED_SOURCE_ARCHIVE_SHA256
            or authority.get("activation_request") != activation.get("activation_request")
            or authority.get("image_runtime_probe")
            != activation.get("image_runtime_probe")
            or authority.get("execution_authorized") is not True
            or authority.get("submission_authorized") is not True
            or authority.get("paper_evidence_adoption_authorized") is not False
            or authority.get("submitted") is not False
            or execution_id in by_id
        ):
            raise ClosureError(f"authorization identity drifted: {execution_id}")
        by_id[execution_id] = dict(row)
        by_id[execution_id]["value"] = authority
    queue_ids = [row[0] for row in expected_ids]
    if list(row["execution_id"] for row in raw_authorizations) != queue_ids:
        raise ClosureError("activation authorization/proc order drifted")
    run_id = target["run_id"]
    authority = by_id[run_id]["value"]
    job = target["job"]
    if (
        authority.get("job_spec_sha256") != job.get("sha256")
        or authority.get("protocol_sha256") != job.get("protocol_sha256")
    ):
        raise ClosureError("fixed proc authorization job/protocol binding drifted")
    return {
        "activation": activation,
        "activation_path": path,
        "request_path": request_path,
        "authorization_binding": {
            key: value for key, value in by_id[run_id].items() if key != "value"
        },
        "authorization": authority,
        "all_authorizations_authenticated": True,
    }


def _archive_relative(proc: int, run_id: str) -> Path:
    return ARCHIVE_DIR_RELATIVE / f"{run_id}__{CLUSTER_ID}__{proc}.tar.gz"


def _identity_relative(proc: int, run_id: str) -> Path:
    return EVIDENCE_DIR_RELATIVE / (
        f"{run_id}__{CLUSTER_ID}__{proc}_remote_archive_identity.json"
    )


def _receipt_relative(proc: int, run_id: str) -> Path:
    return RECEIPT_DIR_RELATIVE / (
        f"paper_i_ra_adapt_page12_cluster{CLUSTER_ID}_proc{proc:02d}_"
        f"{run_id}_closure_receipt_20260813.json"
    )


def _remote_path(proc: int, run_id: str) -> str:
    return f"{REMOTE_OUTPUT_ROOT}/{run_id}__{CLUSTER_ID}__{proc}.tar.gz"


def _hash_stream(stream: BinaryIO) -> tuple[str, int, bytes | None]:
    digest = hashlib.sha256()
    size = 0
    captured: bytes | None = b""
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
        if captured is not None:
            if size <= JSON_CAPTURE_LIMIT_BYTES:
                captured += block
            else:
                captured = None
    return digest.hexdigest(), size, captured


def _captured_json(
    observed: Mapping[str, Mapping[str, Any]],
    relative: str,
    *,
    label: str,
) -> dict[str, Any]:
    row = observed.get(relative)
    if not isinstance(row, Mapping) or row.get("captured") is None:
        raise ClosureError(f"{label} is absent or unexpectedly large")
    try:
        value = json.loads(row["captured"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"{label} must be a JSON object")
    return value


def _authenticate_archive(
    workspace_root: Path,
    proc: int,
    target: Mapping[str, Any],
    activation: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = target["run_id"]
    archive_relative = _archive_relative(proc, run_id)
    archive_path = workspace_root / archive_relative
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ClosureError(f"fixed fetched archive is absent or unsafe: {archive_path}")
    archive_size = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    observed: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                raw = member.name
                while raw.startswith("./"):
                    raw = raw[2:]
                if member.isdir() and raw in {"", "."}:
                    continue
                relative = _safe_relative(member.name, label="archive member")
                if relative in seen:
                    raise ClosureError(f"duplicate archive member: {relative}")
                seen.add(relative)
                if member.issym() or member.islnk():
                    raise ClosureError(f"linked archive member is forbidden: {relative}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ClosureError(f"unsafe archive member type: {relative}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ClosureError(f"unreadable archive member: {relative}")
                digest, size, captured = _hash_stream(stream)
                observed[relative] = {
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": size,
                    "captured": captured,
                }
    except (tarfile.TarError, OSError) as exc:
        raise ClosureError(f"fixed fetched archive is not a readable gzip tar: {archive_path}") from exc

    roots = {"worker_exit_status.txt", "worker_receipt.json"}
    if not roots.issubset(observed):
        raise ClosureError("archive lacks fixed worker root receipts")
    exit_payload = observed["worker_exit_status.txt"]["captured"]
    if exit_payload is None or exit_payload.strip() != b"0":
        raise ClosureError("worker exit status is nonzero or unreadable")
    worker = _captured_json(observed, "worker_receipt.json", label="worker receipt")
    _verify_self_digest(worker, label="worker receipt")
    job = target["job"]
    authority = activation["authorization"]
    rounds = worker.get("controller_rounds_completed")
    if (
        worker.get("schema") != WORKER_SCHEMA
        or worker.get("status") != "passed"
        or worker.get("package_id") != EXPECTED_PACKAGE_ID
        or worker.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or worker.get("execution_id") != run_id
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("authorization_sha256") != authority.get("sha256")
        or type(rounds) is not int
        or not 1 <= rounds <= TARGET_HORIZON
        or worker.get("fresh_start") is not True
    ):
        raise ClosureError("worker receipt identity drifted")

    raw_artifacts = worker.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ClosureError("worker artifact inventory is absent")
    declared: dict[str, Mapping[str, Any]] = {}
    for row in raw_artifacts:
        if not isinstance(row, Mapping):
            raise ClosureError("worker artifact row is invalid")
        relative = _safe_relative(row.get("path"), label="worker artifact")
        if relative in declared or relative in roots:
            raise ClosureError(f"duplicate or reserved worker artifact: {relative}")
        if type(row.get("size_bytes")) is not int or not isinstance(row.get("sha256"), str):
            raise ClosureError(f"worker artifact binding is incomplete: {relative}")
        declared[relative] = row
    if set(observed) != roots | set(declared):
        raise ClosureError("archive contains missing or unbound files")
    for relative, row in declared.items():
        actual = observed[relative]
        if row.get("sha256") != actual["sha256"] or row.get("size_bytes") != actual["size_bytes"]:
            raise ClosureError(f"worker artifact binding drifted: {relative}")

    expected_artifacts = job.get("expected_run_artifacts")
    if not isinstance(expected_artifacts, Mapping):
        raise ClosureError("fixed job expected artifact inventory is absent")
    expected_paths: dict[str, str] = {}
    for role in ("checkpoint", "estimator_ledger", "execution_manifest", "result", "summary"):
        row = expected_artifacts.get(role)
        if (
            not isinstance(row, Mapping)
            or row.get("required") is not True
            or row.get("direct_file_required") is not True
        ):
            raise ClosureError(f"fixed expected artifact contract drifted: {role}")
        expected_paths[role] = _safe_relative(row.get("path"), label=f"expected {role}")
    checkpoint_path = PurePosixPath(expected_paths["checkpoint"])
    sidecars: dict[str, str] = {}
    for role in ("estimator_call_ledger_checkpoint", "verified_singleton_resume"):
        prefix = f"{checkpoint_path.stem}.{role}."
        matches = [
            relative
            for relative in declared
            if PurePosixPath(relative).parent == checkpoint_path.parent
            and PurePosixPath(relative).name.startswith(prefix)
            and PurePosixPath(relative).name.endswith(".json")
        ]
        if len(matches) != 1:
            raise ClosureError(f"worker checkpoint sidecar inventory drifted: {role}")
        relative = matches[0]
        expected_relative = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.{role}.{observed[relative]['sha256'][:16]}.json"
        ).as_posix()
        if relative != expected_relative:
            raise ClosureError(f"worker checkpoint sidecar content address drifted: {role}")
        sidecars[role] = relative
    if set(declared) != set(expected_paths.values()) | set(sidecars.values()):
        raise ClosureError("worker artifact inventory differs from the fixed job")

    manifest_path = expected_paths["execution_manifest"]
    manifest = _captured_json(observed, manifest_path, label="execution manifest")
    _verify_self_digest(manifest, label="execution manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("package_id") != EXPECTED_PACKAGE_ID
        or manifest.get("campaign_id") != EXPECTED_CAMPAIGN_ID
        or manifest.get("execution_id") != run_id
        or manifest.get("job_spec_sha256") != job.get("sha256")
        or manifest.get("authorization_sha256") != authority.get("sha256")
        or manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or manifest.get("route_contract_sha256") != job.get("route_contract_sha256")
        or manifest.get("comparator_policy") != job.get("comparator_policy")
        or manifest.get("target_horizon") != TARGET_HORIZON
        or manifest.get("controller_rounds_completed") != rounds
        or manifest.get("fresh_start") is not True
        or manifest.get("source_checkpoint_consumed") is not False
        or worker.get("execution_manifest_sha256") != manifest.get("sha256")
    ):
        raise ClosureError("execution manifest identity drifted")
    output_payloads = manifest.get("output_payloads")
    expected_output_roles = {"checkpoint", "estimator_ledger", "result", "summary"}
    if not isinstance(output_payloads, Mapping) or set(output_payloads) != expected_output_roles:
        raise ClosureError("execution manifest output inventory drifted")
    for role in sorted(expected_output_roles):
        row = output_payloads.get(role)
        relative = expected_paths[role]
        actual = observed[relative]
        if (
            not isinstance(row, Mapping)
            or row.get("path") != relative
            or row.get("sha256") != actual["sha256"]
            or row.get("size_bytes") != actual["size_bytes"]
        ):
            raise ClosureError(f"execution manifest output binding drifted: {role}")
    summary = _captured_json(observed, expected_paths["summary"], label="Paper-I summary")
    if summary.get("schema") != "paper_i_run_summary_v1":
        raise ClosureError("Paper-I summary schema drifted")
    inventory = [
        {key: row[key] for key in ("path", "sha256", "size_bytes")}
        for row in sorted(observed.values(), key=lambda item: item["path"])
    ]
    return {
        "path": archive_path,
        "relative": archive_relative,
        "sha256": archive_sha,
        "size_bytes": archive_size,
        "inventory": inventory,
        "worker": worker,
        "manifest": manifest,
        "summary_path": expected_paths["summary"],
        "summary_binding": {
            key: observed[expected_paths["summary"]][key]
            for key in ("path", "sha256", "size_bytes")
        },
        "artifact_count": len(declared),
        "regular_member_count": len(observed),
    }


def _authenticate_remote_identity(
    workspace_root: Path,
    proc: int,
    target: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = target["run_id"]
    relative = _identity_relative(proc, run_id)
    path = workspace_root / relative
    identity = _load_object(path, label="remote/local archive identity evidence")
    _verify_self_digest(identity, label="remote/local archive identity evidence")
    if (
        identity.get("schema")
        != "paper_i_ra_adapt_page12_insertion_comparator_remote_archive_identity_v1"
        or identity.get("status")
        != "passed_remote_local_size_sha256_match_after_atomic_rename"
        or identity.get("cluster_id") != CLUSTER_ID
        or identity.get("proc_id") != proc
        or identity.get("execution_id") != run_id
        or identity.get("remote_path") != _remote_path(proc, run_id)
        or identity.get("local_path") != archive["relative"].as_posix()
        or identity.get("remote_size_bytes") != archive["size_bytes"]
        or identity.get("local_size_bytes") != archive["size_bytes"]
        or identity.get("remote_sha256") != archive["sha256"]
        or identity.get("local_sha256") != archive["sha256"]
        or identity.get("gzip_integrity_passed") is not True
        or identity.get("tar_readability_passed") is not True
        or identity.get("atomic_local_rename_completed") is not True
        or identity.get("remote_state")
        != "preserved_after_exact_size_sha256_verified_fetch"
        or not isinstance(identity.get("captured_at_utc"), str)
    ):
        raise ClosureError("remote/local archive identity evidence drifted")
    return {"path": path, "relative": relative, "value": identity}


def _authenticate_submission_receipt() -> dict[str, Any]:
    if (
        not SUBMISSION_RECEIPT_PATH.is_file()
        or SUBMISSION_RECEIPT_PATH.is_symlink()
        or _sha256_file(SUBMISSION_RECEIPT_PATH) != EXPECTED_SUBMISSION_RECEIPT_SHA256
    ):
        raise ClosureError("fixed all-24 submission receipt drifted")
    receipt = _load_object(SUBMISSION_RECEIPT_PATH, label="all-24 submission receipt")
    cluster_rows = receipt.get("clusters")
    page12 = receipt.get("submission_inputs", {}).get("page12")
    cluster = next(
        (
            row
            for row in cluster_rows
            if isinstance(row, Mapping) and row.get("cluster_id") == CLUSTER_ID
        ),
        None,
    ) if isinstance(cluster_rows, list) else None
    if (
        not isinstance(cluster, Mapping)
        or cluster.get("submitted_procs") != ROW_COUNT
        or cluster.get("batch_name") != "paper-i-page12-insertion-all-20260812"
        or not isinstance(page12, Mapping)
        or page12.get("package_manifest_canonical_sha256")
        != EXPECTED_PACKAGE_CANONICAL_SHA256
        or page12.get("source_archive_sha256") != EXPECTED_SOURCE_ARCHIVE_SHA256
        or page12.get("activation_manifest_canonical_sha256")
        != EXPECTED_ACTIVATION_CANONICAL_SHA256
        or page12.get("queue_rows") != ROW_COUNT
        or page12.get("queue_sha256") != EXPECTED_QUEUE_SHA256
    ):
        raise ClosureError("all-24 Page-12 submission identity drifted")
    return receipt


def _authenticate_all(workspace_root: Path, proc: int) -> dict[str, Any]:
    target = _load_package_and_target(proc)
    activation = _load_activation(target)
    _authenticate_submission_receipt()
    archive = _authenticate_archive(workspace_root, proc, target, activation)
    identity = _authenticate_remote_identity(
        workspace_root, proc, target, archive
    )
    return {
        "target": target,
        "activation": activation,
        "archive": archive,
        "identity": identity,
    }


def _file_binding(path: Path, *, root: Path, canonical: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical is not None:
        row["canonical_sha256"] = canonical
    return row


def _preflight(workspace_root: Path, proc: int) -> dict[str, Any]:
    closed = _authenticate_all(workspace_root, proc)
    target = closed["target"]
    archive = closed["archive"]
    return {
        "schema": (
            "paper_i_ra_adapt_page12_insertion_comparator_"
            "closure_preflight_v1"
        ),
        "status": "passed_ready_to_finalize",
        "cluster_id": CLUSTER_ID,
        "proc_id": proc,
        "run_id": target["run_id"],
        "regime_id": target["job"]["regime_id"],
        "comparator_policy": target["job"]["comparator_policy"],
        "controller_rounds_completed": archive["worker"]["controller_rounds_completed"],
        "archive": {
            "path": archive["relative"].as_posix(),
            "sha256": archive["sha256"],
            "size_bytes": archive["size_bytes"],
            "regular_member_count": archive["regular_member_count"],
            "declared_artifact_count": archive["artifact_count"],
        },
        "worker_receipt_canonical_sha256": archive["worker"]["sha256"],
        "execution_manifest_canonical_sha256": archive["manifest"]["sha256"],
        "summary_json": {"path_inside_archive": archive["summary_path"]},
        "authentication_checks": {
            "sealed_package_identity_passed": True,
            "source_identity_passed": True,
            "job_identity_passed": True,
            "protocol_identity_passed": True,
            "route_identity_passed": True,
            "insertion_identity_passed": True,
            "activation_manifest_and_all_authorizations_passed": True,
            "remote_local_identity_passed": True,
            "full_member_inventory_hash_size_closure_passed": True,
        },
        "writes_performed": False,
        "network_performed": False,
        "scheduler_mutation_performed": False,
        "scientific_execution_performed": False,
    }


def _atomic_publish(path: Path, receipt: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ClosureError(f"fixed closure receipt already exists; refusing overwrite: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ClosureError(f"closure receipt parent is absent or unsafe: {parent}")
    temporary = parent / f".proc-receipt.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    check = _load_object(temporary, label="temporary closure receipt")
    _verify_self_digest(check, label="temporary closure receipt")
    if check != receipt:
        raise ClosureError("temporary closure receipt bytes drifted")
    os.replace(temporary, path)


def _finalize(workspace_root: Path, proc: int) -> dict[str, Any]:
    closed = _authenticate_all(workspace_root, proc)
    target = closed["target"]
    activation = closed["activation"]
    archive = closed["archive"]
    identity = closed["identity"]
    job = target["job"]
    protocol = target["protocol"]
    run_id = target["run_id"]
    receipt_relative = _receipt_relative(proc, run_id)
    receipt_path = workspace_root / receipt_relative
    receipt = _digested(
        {
            "schema": CLOSURE_RECEIPT_SCHEMA,
            "status": CLOSURE_RECEIPT_STATUS,
            "created_at_utc": identity["value"]["captured_at_utc"],
            "cluster_id": CLUSTER_ID,
            "proc_id": proc,
            "run_id": run_id,
            "regime_id": job["regime_id"],
            "nph": job["nph"],
            "comparator_policy": job["comparator_policy"],
            "typed_insertion_kind": job["typed_insertion_kind"],
            "runtime_insertion_mode": job["runtime_insertion_mode"],
            "controller_rounds_completed": archive["worker"]["controller_rounds_completed"],
            "target_horizon": TARGET_HORIZON,
            "package_manifest": _file_binding(
                target["package_path"],
                root=SOURCE_REPO_ROOT,
                canonical=target["package"]["sha256"],
            ),
            "source_archive": target["package"]["source_archive"],
            "job": _file_binding(
                target["job_path"],
                root=SOURCE_REPO_ROOT,
                canonical=job["sha256"],
            ),
            "protocol": _file_binding(
                target["protocol_path"],
                root=SOURCE_REPO_ROOT,
                canonical=protocol["sha256"],
            ),
            "route_contract_canonical_sha256": job["route_contract_sha256"],
            "activation_manifest": _file_binding(
                activation["activation_path"],
                root=SOURCE_REPO_ROOT,
                canonical=activation["activation"]["sha256"],
            ),
            "authorization": {
                **activation["authorization_binding"],
                "path": (
                    ACTIVATION_RELATIVE
                    / activation["authorization_binding"]["path"]
                ).as_posix(),
            },
            "remote_local_identity_evidence": _file_binding(
                identity["path"],
                root=workspace_root,
                canonical=identity["value"]["sha256"],
            ),
            "archive": {
                "path": archive["relative"].as_posix(),
                "remote_path": _remote_path(proc, run_id),
                "sha256": archive["sha256"],
                "size_bytes": archive["size_bytes"],
                "regular_member_count": archive["regular_member_count"],
                "declared_artifact_count": archive["artifact_count"],
                "inventory": archive["inventory"],
            },
            "worker_receipt": {
                "path_inside_archive": "worker_receipt.json",
                "schema": archive["worker"]["schema"],
                "status": archive["worker"]["status"],
                "canonical_sha256": archive["worker"]["sha256"],
            },
            "execution_manifest": {
                "path_inside_archive": job["expected_run_artifacts"]["execution_manifest"]["path"],
                "schema": archive["manifest"]["schema"],
                "status": archive["manifest"]["status"],
                "canonical_sha256": archive["manifest"]["sha256"],
            },
            "summary_json": {
                "path_inside_archive": archive["summary_path"],
                "sha256": archive["summary_binding"]["sha256"],
                "size_bytes": archive["summary_binding"]["size_bytes"],
            },
            "authentication_checks": {
                "exact_cluster_proc_mapping_passed": True,
                "sealed_package_identity_passed": True,
                "source_identity_passed": True,
                "job_identity_passed": True,
                "protocol_identity_passed": True,
                "route_identity_passed": True,
                "insertion_identity_passed": True,
                "activation_manifest_passed": True,
                "all_twelve_authorizations_passed": activation["all_authorizations_authenticated"],
                "target_authorization_passed": True,
                "remote_local_identity_passed": True,
                "tar_safety_and_unique_inventory_passed": True,
                "worker_receipt_self_digest_passed": True,
                "execution_manifest_self_digest_passed": True,
                "every_declared_artifact_hash_size_passed": True,
                "full_member_inventory_hash_size_closure_passed": True,
            },
            "paper_evidence_adopted": False,
            "network_performed_by_action": False,
            "scheduler_mutation_performed_by_action": False,
            "scientific_execution_performed_by_action": False,
        }
    )
    _atomic_publish(receipt_path, receipt)
    return {
        "schema": "paper_i_ra_adapt_page12_insertion_comparator_finalize_result_v1",
        "status": "passed_strict_receipt_atomically_published",
        "cluster_id": CLUSTER_ID,
        "proc_id": proc,
        "run_id": run_id,
        "receipt_path": receipt_relative.as_posix(),
        "receipt_canonical_sha256": receipt["sha256"],
        "network_performed": False,
        "scheduler_mutation_performed": False,
        "scientific_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--proc", type=int, choices=range(ROW_COUNT), required=True)
    parser.add_argument("--workspace-root", type=Path, default=SOURCE_REPO_ROOT)
    args = parser.parse_args()
    workspace_root = args.workspace_root.resolve()
    try:
        result = (
            _preflight(workspace_root, args.proc)
            if args.preflight
            else _finalize(workspace_root, args.proc)
        )
    except (ClosureError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
