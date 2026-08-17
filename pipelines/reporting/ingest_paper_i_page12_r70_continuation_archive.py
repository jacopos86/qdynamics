#!/usr/bin/env python3
"""Ingest one authenticated Page-12 round-70 continuation archive.

The continuation result contains the hydrated first 50 accepted rounds and the
new rounds 51--70.  This ingester validates the complete package/job/worker/
manifest/summary chain, proves that the hydrated prefix agrees with the
previously authenticated Page-12 result, and emits a compact typed adapter.

The adapter intentionally preserves the original round-50 Qiskit and
``S_alg`` observation.  Round 70 extends only the scientific error trajectory;
it does not change Paper-I's fixed-round resource-reporting convention.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260811_v1_chtc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260812_page12_strong_r70_continuations_v1"
)
BASE_COMPLETED_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_phase0_completed_20260809"
)

PACKAGE_ID = (
    "paper_i_page12_strong_holstein_r70_accepted_continuations_"
    "20260811_v1_chtc"
)
CAMPAIGN_ID = "paper_i_page12_strong_holstein_r70_continuations_20260811_v1"
PACKAGE_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_chtc_package_v1"
PACKAGE_MANIFEST_SHA256 = (
    "3051aa31402d6c71d87ec7ca9d12006ba95fd95ae22d61ff679110578de2671b"
)
JOB_SCHEMA = "paper_i_page12_strong_r70_chtc_job_v1"
AUTHORIZATION_SCHEMA = "paper_i_page12_strong_r70_chtc_authorization_v1"
ACTIVATION_SCHEMA = "paper_i_page12_strong_r70_chtc_activation_v1"
WORKER_SCHEMA = "paper_i_page12_strong_r70_worker_receipt_v2"
EXECUTION_MANIFEST_SCHEMA = "paper_i_page12_strong_r70_execution_manifest_v2"
SUMMARY_SCHEMA = "paper_i_run_summary_v1"
ROUTE_ID = (
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
)
ROUTE_CONTRACT_SHA256 = (
    "9811652b332b592bee048a8e5f3048972256abae186921ed7efea52bfd5f3dd8"
)
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
COMPILE_CONVENTION = "table_i_basis_gate_transpile_v1"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70

BASE_COMPLETED_ADAPTERS = {
    "weak_strong": "9605157.3_completed_report_adapter.json",
    "intermediate_strong": "9605157.4_completed_report_adapter.json",
    "strong_strong_u8": "9605157.5_completed_report_adapter.json",
}


class ContinuationIngestError(ValueError):
    """The archive cannot support an authenticated Page-12 continuation."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(value)
    if "sha256" in unsigned:
        raise ContinuationIngestError("refusing to digest a pre-digested mapping")
    return {
        **unsigned,
        "sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> None:
    claimed = value.get("sha256")
    unsigned = {key: row for key, row in value.items() if key != "sha256"}
    observed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ContinuationIngestError(f"{label}: self digest drifted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationIngestError(f"{label}: mapping required")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ContinuationIngestError(f"{label}: sequence required")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ContinuationIngestError(f"{label}: integer required")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ContinuationIngestError(f"{label}: integer required") from exc
    if result != value or result < minimum:
        raise ContinuationIngestError(f"{label}: invalid integer")
    return result


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContinuationIngestError(f"{label}: finite scalar required") from exc
    if not math.isfinite(result):
        raise ContinuationIngestError(f"{label}: finite scalar required")
    return result


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationIngestError(f"{label}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContinuationIngestError(f"{label}: JSON object required")
    return value


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationIngestError(f"{label}: unsafe or missing file")
    return _json_object(path.read_bytes(), label=label)


def _safe_relative(raw: Any, *, label: str) -> Path:
    value = str(raw)
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ContinuationIngestError(f"{label}: unsafe relative path")
    return Path(*path.parts)


def _safe_member_name(raw: str) -> str:
    name = str(raw)
    while name.startswith("./"):
        name = name[2:]
    return _safe_relative(name, label="archive member").as_posix()


def _relative_local_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationIngestError(f"unsafe or missing file: {path}")
    return {
        "path": _relative_local_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_bound_file(
    package_dir: Path,
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = package_dir / _safe_relative(row.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != row.get("size_bytes")
        or sha256_file(path) != row.get("sha256")
    ):
        raise ContinuationIngestError(f"{label}: bound file drifted")
    if not canonical:
        return path, None
    value = _load_json(path, label=label)
    verify_self_digest(value, label=label)
    if value.get("sha256") != row.get("canonical_sha256"):
        raise ContinuationIngestError(f"{label}: canonical digest drifted")
    return path, value


def _package_job_for_proc(
    package_dir: Path, proc_id: int
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    """Validate the sealed package and return manifest/job/authorization."""

    proc = _integer(proc_id, label="proc id")
    package_manifest_path = package_dir / "package_manifest.json"
    manifest = _load_json(package_manifest_path, label="package manifest")
    verify_self_digest(manifest, label="package manifest")
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("sha256") != PACKAGE_MANIFEST_SHA256
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or manifest.get("row_count") != 3
        or manifest.get("status")
        != "passed_inert_three_authenticated_continuations"
    ):
        raise ContinuationIngestError("package manifest identity drifted")

    queue_path, _ = _verify_bound_file(
        package_dir, manifest.get("queue"), label="queue"
    )
    rows = [line.split("\t") for line in queue_path.read_text().splitlines()]
    if len(rows) != 3 or proc >= len(rows) or len(rows[proc]) != 12:
        raise ContinuationIngestError(f"proc {proc}: queue row drifted")
    (
        execution_id,
        job_relative,
        protocol_relative,
        authorization_relative,
        resume_relative,
        _resume_manifest_relative,
        _checkpoint_validation_relative,
        resume_sha256,
        *_resources,
    ) = rows[proc]

    job_rows = {
        str(row["execution_id"]): row
        for row in _sequence(manifest.get("jobs"), label="manifest jobs")
    }
    protocol_rows = {
        str(row["execution_id"]): row
        for row in _sequence(manifest.get("protocols"), label="manifest protocols")
    }
    if execution_id not in job_rows or execution_id not in protocol_rows:
        raise ContinuationIngestError("queue execution is absent from package manifest")
    job_path, job = _verify_bound_file(
        package_dir, job_rows[execution_id], label="job spec", canonical=True
    )
    assert job is not None
    if job_path.relative_to(package_dir).as_posix() != job_relative:
        raise ContinuationIngestError("queue/job path binding drifted")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("execution_id") != execution_id
        or job.get("route_id") != ROUTE_ID
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("candidate_representation") != CANDIDATE_REPRESENTATION
        or job.get("source_horizon") != SOURCE_HORIZON
        or job.get("resume_round") != SOURCE_HORIZON
        or job.get("target_horizon") != TARGET_HORIZON
        or _mapping(job.get("resume_archive"), label="job resume archive").get(
            "path"
        )
        != resume_relative
        or _mapping(job.get("resume_archive"), label="job resume archive").get(
            "sha256"
        )
        != resume_sha256
    ):
        raise ContinuationIngestError(f"proc {proc}: job identity drifted")

    protocol_path, protocol = _verify_bound_file(
        package_dir,
        protocol_rows[execution_id],
        label="resolved protocol",
        canonical=True,
    )
    assert protocol is not None
    job_protocol = _mapping(job.get("protocol"), label="job protocol")
    if (
        protocol_path.relative_to(package_dir).as_posix() != protocol_relative
        or dict(job_protocol) != dict(protocol_rows[execution_id])
        or protocol.get("sha256") != job.get("protocol_sha256")
        or _mapping(protocol.get("route_contract"), label="protocol route").get(
            "sha256"
        )
        != ROUTE_CONTRACT_SHA256
    ):
        raise ContinuationIngestError("job/protocol binding drifted")

    activation_path = package_dir / "activation/activation_manifest.json"
    activation = _load_json(activation_path, label="activation manifest")
    verify_self_digest(activation, label="activation manifest")
    if (
        activation.get("schema") != ACTIVATION_SCHEMA
        or activation.get("package_id") != PACKAGE_ID
        or activation.get("campaign_id") != CAMPAIGN_ID
        or activation.get("package_manifest_sha256") != manifest.get("sha256")
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
    ):
        raise ContinuationIngestError("activation manifest identity drifted")
    authorization_rows = {
        str(row["execution_id"]): row
        for row in _sequence(
            activation.get("authorizations"), label="activation authorizations"
        )
    }
    if execution_id not in authorization_rows:
        raise ContinuationIngestError("execution authorization binding is absent")
    authorization_path, authorization = _verify_bound_file(
        package_dir,
        authorization_rows[execution_id],
        label="execution authorization",
        canonical=True,
    )
    assert authorization is not None
    if (
        authorization_path.relative_to(package_dir).as_posix()
        != authorization_relative
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("package_id") != PACKAGE_ID
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("execution_id") != execution_id
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
    ):
        raise ContinuationIngestError("execution authorization identity drifted")
    return manifest, job_path, job, authorization_path, authorization


def _remote_binding(
    *, path: str, sha256: str, size_bytes: Any
) -> dict[str, Any]:
    remote_path = str(path)
    remote_sha = str(sha256).lower()
    if not remote_path or len(remote_sha) != 64 or any(
        value not in "0123456789abcdef" for value in remote_sha
    ):
        raise ContinuationIngestError("remote archive binding is invalid")
    return {
        "path": remote_path,
        "sha256": remote_sha,
        "size_bytes": _integer(size_bytes, label="remote archive size", minimum=1),
    }


def _scan_archive(
    archive_path: Path,
    *,
    expected_paths: set[str] | None,
    retained_json_paths: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    """Hash every regular member and retain only the three small JSON docs."""

    observed: dict[str, dict[str, Any]] = {}
    retained: dict[str, bytes] = {}
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                name = _safe_member_name(member.name)
                if member.isdir():
                    continue
                if member.issym() or member.islnk() or not member.isfile():
                    raise ContinuationIngestError(
                        f"forbidden archive member type: {name}"
                    )
                if name in observed:
                    raise ContinuationIngestError(f"duplicate archive member: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ContinuationIngestError(
                        f"archive member has no bytes: {name}"
                    )
                digest = hashlib.sha256()
                payload = bytearray() if name in retained_json_paths else None
                total = 0
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    total += len(chunk)
                    if payload is not None:
                        payload.extend(chunk)
                if total != member.size:
                    raise ContinuationIngestError(
                        f"archive member was truncated: {name}"
                    )
                observed[name] = {
                    "sha256": digest.hexdigest(),
                    "size_bytes": total,
                }
                if payload is not None:
                    retained[name] = bytes(payload)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ContinuationIngestError(f"full archive scan failed: {exc}") from exc
    if expected_paths is not None and set(observed) != expected_paths:
        missing = sorted(expected_paths - set(observed))
        extra = sorted(set(observed) - expected_paths)
        raise ContinuationIngestError(
            f"archive member closure drifted; missing={missing}, extra={extra}"
        )
    if set(retained) != retained_json_paths:
        raise ContinuationIngestError("archive JSON document closure drifted")
    return observed, retained


def _bound_artifact_map(value: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in _sequence(value, label=label):
        row = _mapping(raw, label=f"{label} row")
        path = str(row.get("path", ""))
        if not path or path in result:
            raise ContinuationIngestError(f"{label}: duplicate or absent path")
        result[path] = row
    return result


def _work(value: Any, *, label: str) -> dict[str, Any]:
    row = _mapping(value, label=label)
    components = _mapping(row.get("components"), label=f"{label} components")
    keys = ("n_h_outer", "n_h_refit", "n_grad", "n_metric")
    if set(components) != set(keys):
        raise ContinuationIngestError(f"{label}: component closure drifted")
    normalized = {
        key: _integer(components[key], label=f"{label}.{key}") for key in keys
    }
    s_alg = _integer(row.get("s_alg"), label=f"{label}.s_alg")
    if s_alg != sum(normalized.values()):
        raise ContinuationIngestError(f"{label}: S_alg does not close")
    return {"components": normalized, "s_alg": s_alg}


def _same_scalar(left: float, right: float) -> bool:
    scale = max(1.0, abs(left), abs(right))
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=max(1.0e-12, 128.0 * math.ulp(scale)),
    )


def _base_completed_adapter(regime_id: str) -> tuple[Path, dict[str, Any]]:
    try:
        name = BASE_COMPLETED_ADAPTERS[regime_id]
    except KeyError as exc:
        raise ContinuationIngestError(
            f"unsupported Page-12 continuation regime: {regime_id}"
        ) from exc
    path = BASE_COMPLETED_DIR / name
    value = _load_json(path, label="base Page-12 completed adapter")
    verify_self_digest(value, label="base Page-12 completed adapter")
    points = _sequence(value.get("points"), label="base accepted points")
    terminal = _mapping(value.get("terminal"), label="base terminal")
    costs = _mapping(terminal.get("costs"), label="base round-50 costs")
    if (
        value.get("schema") != "paper_i_phase0_completed_remote_summary_adapter_v1"
        or value.get("status")
        != "passed_remote_summary_extract_full_archive_preserved"
        or value.get("regime_id") != regime_id
        or value.get("controller_rounds_completed") != SOURCE_HORIZON
        or len(points) != SOURCE_HORIZON
        or [row.get("k") for row in points] != list(range(1, SOURCE_HORIZON + 1))
        or terminal.get("k") != SOURCE_HORIZON
        or set(costs) != {"N2q", "D2q", "Dc", "W1q", "S_alg"}
    ):
        raise ContinuationIngestError("base Page-12 adapter identity drifted")
    return path, value


def build_outputs(
    *,
    archive_path: Path,
    cluster_id: int,
    proc_id: int,
    remote_archive: Mapping[str, Any],
    package_dir: Path = DEFAULT_PACKAGE_DIR,
    retrieved_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the typed continuation adapter and retrieval receipt."""

    cluster = _integer(cluster_id, label="cluster id", minimum=1)
    proc = _integer(proc_id, label="proc id")
    if not archive_path.is_file() or archive_path.is_symlink():
        raise ContinuationIngestError(f"unsafe or missing archive: {archive_path}")
    remote = _remote_binding(
        path=str(remote_archive.get("path", "")),
        sha256=str(remote_archive.get("sha256", "")),
        size_bytes=remote_archive.get("size_bytes"),
    )
    if archive_path.stat().st_size != remote["size_bytes"]:
        raise ContinuationIngestError("local archive size differs from remote identity")
    local_sha256 = sha256_file(archive_path)
    if local_sha256 != remote["sha256"]:
        raise ContinuationIngestError(
            "local archive SHA-256 differs from remote identity"
        )

    (
        package_manifest,
        job_path,
        job,
        authorization_path,
        authorization,
    ) = _package_job_for_proc(package_dir, proc)
    execution_id = str(job["execution_id"])
    expected = {
        str(role): str(path)
        for role, path in _mapping(
            job.get("expected_artifacts"), label="job expected artifacts"
        ).items()
    }
    required_roles = {
        "execution_manifest",
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    }
    if set(expected) != required_roles:
        raise ContinuationIngestError("job expected-artifact closure drifted")
    retained_paths = {
        "worker_receipt.json",
        expected["execution_manifest"],
        expected["summary"],
    }
    observed, retained = _scan_archive(
        archive_path,
        # The worker may publish checkpoint-owned authenticated sidecars in
        # addition to the five job-declared top-level roles.  Their exact
        # names are authoritative only after the self-digested worker receipt
        # is read, so scan every regular member first and close the set below.
        expected_paths=None,
        retained_json_paths=retained_paths,
    )
    worker = _json_object(
        retained["worker_receipt.json"], label="worker receipt"
    )
    execution_manifest = _json_object(
        retained[expected["execution_manifest"]], label="execution manifest"
    )
    summary = _json_object(retained[expected["summary"]], label="run summary")
    verify_self_digest(worker, label="worker receipt")
    verify_self_digest(execution_manifest, label="execution manifest")

    if (
        worker.get("schema") != WORKER_SCHEMA
        or worker.get("status") != "passed"
        or worker.get("package_id") != PACKAGE_ID
        or worker.get("campaign_id") != CAMPAIGN_ID
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("authorization_sha256") != authorization.get("sha256")
        or worker.get("execution_manifest_sha256")
        != execution_manifest.get("sha256")
        or worker.get("resume_round") != SOURCE_HORIZON
        or worker.get("controller_rounds_completed") != TARGET_HORIZON
        or worker.get("accepted_state_resume") is not True
    ):
        raise ContinuationIngestError("worker receipt identity drifted")
    worker_artifacts = _bound_artifact_map(
        worker.get("artifacts"), label="worker artifacts"
    )
    if not set(expected.values()).issubset(worker_artifacts):
        raise ContinuationIngestError("worker artifact path closure drifted")
    if set(observed) != {"worker_receipt.json", *worker_artifacts}:
        raise ContinuationIngestError("archive/worker artifact closure drifted")
    for path, row in worker_artifacts.items():
        if (
            row.get("sha256") != observed[path]["sha256"]
            or row.get("size_bytes") != observed[path]["size_bytes"]
        ):
            raise ContinuationIngestError(f"worker artifact bytes drifted: {path}")

    output_payloads = _mapping(
        execution_manifest.get("output_payloads"), label="manifest payloads"
    )
    payload_roles = required_roles - {"execution_manifest"}
    preservation = _mapping(
        execution_manifest.get("accepted_prefix_preservation"),
        label="accepted prefix preservation",
    )
    if (
        execution_manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or execution_manifest.get("status") != "passed"
        or execution_manifest.get("package_id") != PACKAGE_ID
        or execution_manifest.get("campaign_id") != CAMPAIGN_ID
        or execution_manifest.get("execution_id") != execution_id
        or execution_manifest.get("job_spec_sha256") != job.get("sha256")
        or execution_manifest.get("authorization_sha256")
        != authorization.get("sha256")
        or execution_manifest.get("protocol_sha256") != job.get("protocol_sha256")
        or execution_manifest.get("route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
        or execution_manifest.get("resume_round") != SOURCE_HORIZON
        or execution_manifest.get("target_horizon") != TARGET_HORIZON
        or execution_manifest.get("controller_rounds_completed") != TARGET_HORIZON
        or execution_manifest.get("accepted_state_resume") is not True
        or preservation.get("status") != "passed"
        or preservation.get("source_round") != SOURCE_HORIZON
        or set(output_payloads) != payload_roles
    ):
        raise ContinuationIngestError("execution manifest identity drifted")
    for role in payload_roles:
        row = _mapping(output_payloads[role], label=f"manifest {role}")
        path = expected[role]
        if (
            row.get("path") != path
            or row.get("sha256") != observed[path]["sha256"]
            or row.get("size_bytes") != observed[path]["size_bytes"]
        ):
            raise ContinuationIngestError(f"manifest payload bytes drifted: {role}")

    provenance = _mapping(summary.get("provenance"), label="summary provenance")
    trace = _sequence(summary.get("accepted_error_trace"), label="accepted trace")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("available_controller_rounds") != TARGET_HORIZON
        or len(trace) != TARGET_HORIZON
        or provenance.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or provenance.get("candidate_representation")
        != CANDIDATE_REPRESENTATION
        or provenance.get("qiskit_compile_convention") != COMPILE_CONVENTION
    ):
        raise ContinuationIngestError("run summary identity drifted")
    exact = _finite(
        provenance.get("exact_same_cutoff_energy"), label="same-cutoff exact energy"
    )
    points: list[dict[str, Any]] = []
    for expected_round, raw in enumerate(trace, 1):
        row = _mapping(raw, label=f"accepted trace row {expected_round}")
        energy = _finite(row.get("accepted_energy"), label="accepted energy")
        error = _finite(row.get("absolute_energy_error"), label="accepted error")
        if (
            row.get("controller_round") != expected_round
            or not _same_scalar(
                _finite(row.get("exact_same_cutoff_energy"), label="trace exact"),
                exact,
            )
            or not _same_scalar(error, abs(energy - exact))
        ):
            raise ContinuationIngestError("accepted 70-point trajectory drifted")
        points.append({"k": expected_round, "energy": energy, "error": error})
    continuation_work = _work(
        summary.get("canonical_all_work"), label="round-70 canonical all work"
    )

    base_path, base = _base_completed_adapter(str(job["regime_id"]))
    if not _same_scalar(
        _finite(base.get("exact_same_cutoff_energy"), label="base exact"), exact
    ):
        raise ContinuationIngestError("continuation/base exact reference drifted")
    base_points = _sequence(base.get("points"), label="base points")
    for expected_round, (base_row, continued_row) in enumerate(
        zip(base_points, points[:SOURCE_HORIZON], strict=True), 1
    ):
        if (
            base_row.get("k") != expected_round
            or not _same_scalar(
                _finite(base_row.get("energy"), label="base energy"),
                continued_row["energy"],
            )
            or not _same_scalar(
                _finite(base_row.get("error"), label="base error"),
                continued_row["error"],
            )
        ):
            raise ContinuationIngestError(
                f"authenticated first-50 trajectory drifted at round {expected_round}"
            )

    base_terminal = _mapping(base.get("terminal"), label="base terminal")
    fixed_costs = {
        key: _integer(value, label=f"base cost {key}")
        for key, value in _mapping(
            base_terminal.get("costs"), label="base round-50 costs"
        ).items()
    }
    merged_points = [copy.deepcopy(dict(row)) for row in base_points] + [
        copy.deepcopy(row) for row in points[SOURCE_HORIZON:]
    ]
    terminal = merged_points[-1]

    def member_source(relative: str) -> dict[str, Any]:
        return {
            "path": f"{remote['path']}::{relative}",
            **observed[relative],
        }

    adapter = digested(
        {
            "schema": "paper_i_page12_r70_continuation_adapter_v1",
            "status": "passed_authenticated_round70_continuation",
            "cluster_id": cluster,
            "proc_id": proc,
            "execution_id": execution_id,
            "regime_id": str(job["regime_id"]),
            "nph": _integer(job.get("nph"), label="job nph", minimum=1),
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "exact_same_cutoff_energy": exact,
            "continuation_points": [
                copy.deepcopy(row) for row in points[SOURCE_HORIZON:]
            ],
            "merged_points": merged_points,
            "latest": copy.deepcopy(terminal),
            "fixed_round_50_reporting": {
                "controller_round": SOURCE_HORIZON,
                "costs": fixed_costs,
                "compile": copy.deepcopy(base_terminal.get("compile")),
                "work_components": copy.deepcopy(
                    base_terminal.get("work_components")
                ),
                "policy": "preserved_authenticated_page12_round50_resources_v1",
            },
            "continuation_terminal": {
                "controller_round": TARGET_HORIZON,
                "energy": terminal["energy"],
                "error": terminal["error"],
                "canonical_all_work": continuation_work,
                "resource_reporting_status": (
                    "not_substituted_for_fixed_round50_reporting"
                ),
            },
            "source": {
                "full_archive": dict(remote),
                "local_archive": {
                    "path": _relative_local_path(archive_path),
                    "sha256": local_sha256,
                    "size_bytes": archive_path.stat().st_size,
                },
                "package_manifest": {
                    **_file_binding(package_dir / "package_manifest.json"),
                    "canonical_sha256": package_manifest["sha256"],
                },
                "job_spec": {
                    **_file_binding(job_path),
                    "canonical_sha256": job["sha256"],
                },
                "execution_authorization": {
                    **_file_binding(authorization_path),
                    "canonical_sha256": authorization["sha256"],
                },
                "base_completed_adapter": {
                    **_file_binding(base_path),
                    "canonical_sha256": base["sha256"],
                },
                "worker_receipt": {
                    **member_source("worker_receipt.json"),
                    "canonical_sha256": worker["sha256"],
                },
                "execution_manifest": {
                    **member_source(expected["execution_manifest"]),
                    "canonical_sha256": execution_manifest["sha256"],
                },
                "summary": member_source(expected["summary"]),
            },
            "reporting_policy": {
                "trajectory": "authenticated_rounds_1_to_50_plus_continuation_51_to_70",
                "qiskit_and_s_alg": "fixed_authenticated_controller_round_50",
                "paper_evidence_adopted": False,
            },
        }
    )
    timestamp = retrieved_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    receipt = digested(
        {
            "schema": "paper_i_page12_r70_verified_retrieval_receipt_v1",
            "status": "passed",
            "cluster_id": cluster,
            "proc_id": proc,
            "execution_id": execution_id,
            "retrieved_utc": str(timestamp),
            "byte_identity_passed": True,
            "archive_member_closure_passed": True,
            "package_job_worker_manifest_summary_closure_passed": True,
            "authenticated_first_50_prefix_passed": True,
            "local_archive": {
                "path": _relative_local_path(archive_path),
                "sha256": local_sha256,
                "size_bytes": archive_path.stat().st_size,
                "regular_member_count": len(observed),
            },
            "remote_archive": dict(remote),
            "adapter_sha256": adapter["sha256"],
        }
    )
    return adapter, receipt


def write_outputs(
    *,
    adapter: Mapping[str, Any],
    receipt: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(adapter['cluster_id'])}.{int(adapter['proc_id'])}"
    adapter_path = output_dir / f"{stem}_page12_r70_continuation_adapter.json"
    receipt_path = output_dir / f"{stem}_retrieval_receipt.json"
    for path in (adapter_path, receipt_path):
        if path.exists() or path.is_symlink():
            raise ContinuationIngestError(
                f"refusing to overwrite completed output: {path}"
            )
    for path, value in ((adapter_path, adapter), (receipt_path, receipt)):
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
    return adapter_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--proc-id", type=int, required=True)
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--remote-sha256", required=True)
    parser.add_argument("--remote-size-bytes", type=int, required=True)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retrieved-utc")
    args = parser.parse_args()
    try:
        adapter, receipt = build_outputs(
            archive_path=args.archive.resolve(),
            cluster_id=args.cluster_id,
            proc_id=args.proc_id,
            remote_archive={
                "path": args.remote_path,
                "sha256": args.remote_sha256,
                "size_bytes": args.remote_size_bytes,
            },
            package_dir=args.package_dir.resolve(),
            retrieved_utc=args.retrieved_utc,
        )
        adapter_path, receipt_path = write_outputs(
            adapter=adapter,
            receipt=receipt,
            output_dir=args.output_dir.resolve(),
        )
    except (ContinuationIngestError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "adapter": str(adapter_path),
                "retrieval_receipt": str(receipt_path),
                "adapter_sha256": adapter["sha256"],
                "retrieval_receipt_sha256": receipt["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
