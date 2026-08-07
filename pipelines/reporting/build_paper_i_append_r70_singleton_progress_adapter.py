#!/usr/bin/env python3
"""Build an authenticated compact adapter for fresh Append-ADAPT R70 cells.

The fetched CHTC archives contain multi-gigabyte result and estimator-ledger
members.  This reporting adapter makes one streaming pass over each exact
archive, retaining only the signed reconstruction checkpoint, typed summary,
and their small authority receipts. Receipt-backed full archives retain their
remote/local transport authentication. A direct full archive can instead be
fully streamed against embedded and local sealed authority, while a compact
diagnostic archive authenticates only its retained members and records the
operator-observed remote full-archive identity as not locally reauthenticated.

Round 50 remains the canonical Paper-I comparison point.  Round 70 is emitted
as a clearly separate diagnostic extension.  Both prefixes cross the same
source-locked Paper-I Qiskit compiler boundary in an isolated subprocess.
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
import re
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc"
)
PACKAGE_ID = (
    "paper_i_append_adapt_stationary_core12_r70_fresh_20260731_v1_chtc"
)
CAMPAIGN_ID = "paper_i_append_adapt_stationary_core_r70_fresh_v1"
CLUSTER_ID = 9_398_375
SOURCE_ARCHIVE_SHA256 = (
    "1f949b0cc8b61dca63911832e8dc8bb32614174755ac476827956bb0812accee"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
PACKAGE_MANIFEST_FILE_SHA256 = (
    "334fb630c061d205d61554dca4b3e4f734edf5af394eda1ddb9c994869efefee"
)
PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "eea38b59e60d727281dc3bdaf6d2efa7880f3f49375ce49e61134fbb35a566ea"
)
ACTIVATION_DIR = PACKAGE_DIR.parent / (
    f"{PACKAGE_ID}_activation_ordinary_held_v1"
)
AUTHORIZATION_FILE_SHA256 = (
    "4662e0b4732559b8efe26c1c27a9be3f94b3649751b6100dd186b5b55ca8d979"
)
AUTHORIZATION_CANONICAL_SHA256 = (
    "6e3f7378fdbe352e29ba60c119ae94b3a8351a4687ea5346ec6d5368e1d8fbd4"
)
ACTIVATION_FILE_SHA256 = (
    "77c813fafc0173af32802a8dac055f0e38bb4d38c196065d37c9776d07a5f9b9"
)
ACTIVATION_CANONICAL_SHA256 = (
    "543ff80423e3348800d975bab139ccac53fc87192ef3065e3e9e0fbebd183ce2"
)
RETRIEVAL_RECEIPT_SCHEMA = (
    "paper_i_append_adapt_stationary_core_r70_"
    "retrieval_authentication_receipt_v1"
)
ADAPTER_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
ED_REFERENCE = REPO_ROOT / (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_ed_cutoff_reference_six_regime_20260727.json"
)
ED_REFERENCE_SHA256 = (
    "66a6409790affffd6ce8928d7fb46cc945b57d50e210d3cb215e8039a63c5573"
)

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
COMPLETED_REGIMES = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
PENDING_REGIMES: tuple[str, ...] = ()
REGIME_DISPLAY = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
ED_NAME = {
    "weak_weak": "weak-weak",
    "intermediate_weak": "intermediate-weak",
    "strong_weak_u8": "strong-weak",
    "weak_strong": "weak-strong",
    "intermediate_strong": "intermediate-strong",
    "strong_strong_u8": "strong-strong",
}
NPH_BY_REGIME = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
PROC_BY_REGIME = {
    "weak_weak": 1,
    "intermediate_weak": 3,
    "strong_weak_u8": 5,
    "weak_strong": 7,
    "intermediate_strong": 9,
    "strong_strong_u8": 11,
}

AUTHORITY_MEMBERS = {
    "authority/job.json",
    "authority/execution_authorization.json",
    "authority/activation_manifest.json",
}
COMPACT_MEMBERS = {
    *AUTHORITY_MEMBERS,
    "worker_attempt_receipt.json",
    "worker_outputs/attempt_identity.tsv",
    "worker_outputs/worker_exit_status.txt",
    "worker_outputs/worker_receipt.json",
    "worker_outputs/payload/checkpoint.json",
    "worker_outputs/payload/execution_manifest.json",
    "worker_outputs/payload/summary.json",
}
REQUIRED_WORKER_PATHS = {
    "attempt_identity.tsv",
    "worker_exit_status.txt",
    "worker_receipt.json",
    "payload/checkpoint.json",
    "payload/estimator_ledger.json",
    "payload/execution_manifest.json",
    "payload/result.json",
    "payload/summary.json",
}
EXPECTED_ARCHIVE_MEMBERS = {
    *AUTHORITY_MEMBERS,
    "worker_attempt_receipt.json",
    *(f"worker_outputs/{path}" for path in REQUIRED_WORKER_PATHS),
}
MAX_COMPACT_MEMBER_BYTES = 16 * 1024 * 1024
COMPILE_REQUEST_SCHEMA = "paper_i_append_adapt_r70_compile_request_v1"
COMPILE_RESPONSE_SCHEMA = "paper_i_append_adapt_r70_compile_response_v1"
COST_FIELDS = ("N2q", "D2q", "Dc", "W1q", "S_alg")
SOURCE_MODE_RECEIPT_FULL = "retrieval_receipt_full_archive_v1"
SOURCE_MODE_DIRECT_FULL = "direct_full_archive_sealed_authority_v1"
SOURCE_MODE_COMPACT = "compact_retained_members_remote_observation_v1"
GIANT_WORKER_PATHS = {
    "payload/estimator_ledger.json",
    "payload/result.json",
}


class AdapterInputError(ValueError):
    """Raised when an R70 archive cannot enter the reporting adapter."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "sha256" in result:
        raise AdapterInputError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise AdapterInputError(f"{label} self digest drifted")
    return str(observed)


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AdapterInputError(f"JSON object duplicates key {key!r}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AdapterInputError(f"{label} contains non-finite {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterInputError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise AdapterInputError(f"{label} must be a JSON object")
    return value


def _plain_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().absolute()
    if not resolved.is_file() or resolved.is_symlink():
        raise AdapterInputError(f"{label} is missing, unsafe, or a symlink: {path}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = _plain_file(path, label=label)
    payload = resolved.read_bytes()
    return _load_json_bytes(payload, label=label), payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterInputError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise AdapterInputError(f"{label} must be a sequence")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AdapterInputError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterInputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdapterInputError(f"{label} must be finite")
    return result


def _safe_member_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise AdapterInputError(f"unsafe archive member: {value!r}")
    return value


def _safe_repo_binding(binding: Mapping[str, Any], *, label: str) -> tuple[Path, bytes, dict[str, Any]]:
    relative = PurePosixPath(str(binding.get("path", "")))
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise AdapterInputError(f"{label} path is unsafe")
    path = _plain_file(REPO_ROOT.joinpath(*relative.parts), label=label)
    payload = path.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest() != binding.get("sha256")
        or len(payload) != _integer(binding.get("size_bytes"), label=f"{label} size", minimum=1)
    ):
        raise AdapterInputError(f"{label} byte binding drifted")
    value = _load_json_bytes(payload, label=label)
    canonical = verify_self_digest(value, label=label)
    if binding.get("canonical_sha256") != canonical:
        raise AdapterInputError(f"{label} canonical binding drifted")
    return path, payload, value


def _consume_member(
    stream: BinaryIO,
    *,
    size: int,
    capture: bool,
    label: str,
) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    remaining = size
    blocks: list[bytes] = []
    while remaining:
        block = stream.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise AdapterInputError(f"{label} ended early")
        remaining -= len(block)
        digest.update(block)
        if capture:
            blocks.append(block)
    if stream.read(1):
        raise AdapterInputError(f"{label} exceeded its declared size")
    return digest.hexdigest(), b"".join(blocks) if capture else None


def _stream_compact_members(
    archive: Path,
    *,
    expected_members: set[str] | frozenset[str] = EXPECTED_ARCHIVE_MEMBERS,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    """Fully stream the tar while retaining only bounded compact members."""

    captured: dict[str, bytes] = {}
    observed: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    try:
        with tarfile.open(archive, mode="r|gz") as bundle:
            for member in bundle:
                name = _safe_member_name(member.name)
                if name in names:
                    raise AdapterInputError(f"duplicate archive member: {name}")
                names.add(name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise AdapterInputError(f"non-regular archive member: {name}")
                capture = name in COMPACT_MEMBERS
                if capture and member.size > MAX_COMPACT_MEMBER_BYTES:
                    raise AdapterInputError(f"compact member is unexpectedly large: {name}")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise AdapterInputError(f"archive member is unreadable: {name}")
                digest, payload = _consume_member(
                    stream,
                    size=member.size,
                    capture=capture,
                    label=f"archive member {name}",
                )
                observed[name] = {"sha256": digest, "size_bytes": member.size}
                if payload is not None:
                    captured[name] = payload
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise AdapterInputError(f"archive is not a complete gzip tar: {archive}") from exc
    if names != set(expected_members):
        raise AdapterInputError("archive member closure drifted")
    if set(captured) != COMPACT_MEMBERS:
        raise AdapterInputError("compact archive member closure drifted")
    return captured, observed


def _worker_inventory(value: Any) -> dict[str, dict[str, Any]]:
    rows = _sequence(value, label="worker inventory")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label="worker inventory row")
        path = _safe_member_name(str(row.get("path", "")))
        if path in result:
            raise AdapterInputError("worker inventory duplicates a path")
        digest = str(row.get("sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AdapterInputError("worker inventory digest is malformed")
        result[path] = {
            "sha256": digest,
            "size_bytes": _integer(row.get("size_bytes"), label="worker file size"),
        }
    return result


def _artifact_inventory(value: Any) -> dict[str, dict[str, Any]]:
    rows = _sequence(value, label="worker artifacts")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, label="worker artifact")
        role = str(row.get("role", ""))
        if role in result:
            raise AdapterInputError("worker artifacts duplicate a role")
        result[role] = dict(row)
    if set(result) != {"checkpoint", "estimator_ledger", "execution_manifest", "result", "summary"}:
        raise AdapterInputError("worker artifact role closure drifted")
    return result


def _execution_id(regime: str) -> str:
    return f"r70_fresh__{regime}__nph{NPH_BY_REGIME[regime]}__append_singleton"


def _validate_archive_identity(
    *,
    execution: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    archive_name: str,
    archive_sha256: str,
    archive_size_bytes: int,
) -> str:
    """Close the receipt-to-local-archive identity and return its regime."""

    execution_id = str(execution.get("execution_id", ""))
    matched = re.fullmatch(
        r"r70_fresh__(.+)__nph(3|7)__append_singleton", execution_id
    )
    regime = "" if matched is None else matched.group(1)
    if (
        regime not in COMPLETED_REGIMES
        or execution_id != _execution_id(regime)
        or execution.get("cluster_id") != CLUSTER_ID
        or execution.get("proc_id") != PROC_BY_REGIME[regime]
        or execution.get("attempt_ordinal") != 1
        or execution.get("source_horizon") != 50
        or execution.get("target_horizon") != 70
        or execution.get("fresh_start") is not True
        or execution.get("resume_claimed") is not False
    ):
        raise AdapterInputError("retrieval execution identity drifted")
    local_archive = _mapping(
        retrieval.get("local_archive"), label="receipt local archive"
    )
    expected_name = (
        f"{execution_id}__cluster_{CLUSTER_ID}__proc_"
        f"{PROC_BY_REGIME[regime]}.tar.gz"
    )
    if (
        archive_name != expected_name
        or local_archive.get("sha256") != archive_sha256
        or local_archive.get("size_bytes") != archive_size_bytes
        or retrieval.get("remote_archive_sha256") != archive_sha256
        or retrieval.get("remote_archive_size_bytes") != archive_size_bytes
        or retrieval.get("remote_local_hash_size_match") is not True
        or retrieval.get("expected_final_basename_match") is not True
    ):
        raise AdapterInputError("archive/retrieval-receipt identity drifted")
    return regime


def _ed_reference() -> tuple[dict[str, float], dict[str, Any]]:
    reference, payload = _load_json_file(ED_REFERENCE, label="same-cutoff ED reference")
    if hashlib.sha256(payload).hexdigest() != ED_REFERENCE_SHA256:
        raise AdapterInputError("same-cutoff ED reference byte hash drifted")
    values: dict[str, float] = {}
    rows = _sequence(reference.get("regimes"), label="ED regimes")
    by_name = {str(_mapping(row, label="ED regime").get("name")): row for row in rows}
    for regime in REGIME_ORDER:
        row = _mapping(by_name.get(ED_NAME[regime]), label=f"ED regime {regime}")
        nph = NPH_BY_REGIME[regime]
        if row.get("working_cutoff") != nph:
            raise AdapterInputError(f"{regime}: ED working cutoff drifted")
        cells = {
            _integer(_mapping(cell, label="ED cell").get("M"), label="ED M"): cell
            for cell in _sequence(row.get("cells"), label="ED cells")
        }
        values[regime] = _finite(
            _mapping(cells.get(nph), label=f"{regime} same-cutoff cell").get("E_ED"),
            label=f"{regime} E_ED",
        )
    return values, {
        "path": str(ED_REFERENCE.relative_to(REPO_ROOT)),
        "sha256": ED_REFERENCE_SHA256,
        "schema": reference.get("schema"),
        "cutoff_rule": "job_nph_equals_working_cutoff_v1",
    }


def _fixed_json_file(
    path: Path,
    *,
    label: str,
    expected_file_sha256: str | None = None,
    expected_canonical_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = _plain_file(path, label=label)
    value, payload = _load_json_file(resolved, label=label)
    canonical = verify_self_digest(value, label=label)
    if (
        expected_file_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_file_sha256
    ):
        raise AdapterInputError(f"{label} file digest drifted")
    if (
        expected_canonical_sha256 is not None
        and canonical != expected_canonical_sha256
    ):
        raise AdapterInputError(f"{label} canonical digest drifted")
    return resolved, payload, value


def _regime_from_execution_id(execution_id: str) -> str:
    matched = re.fullmatch(
        r"r70_fresh__(.+)__nph(3|7)__append_singleton", execution_id
    )
    regime = "" if matched is None else matched.group(1)
    if regime not in COMPLETED_REGIMES or execution_id != _execution_id(regime):
        raise AdapterInputError("archive execution is not an expected singleton cell")
    return regime


def _local_authority(
    *,
    execution_id: str,
    regime: str,
    receipt_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_paths = {
        "package": PACKAGE_DIR / "package_manifest.json",
        "job": PACKAGE_DIR / "jobs" / f"{execution_id}.json",
        "authorization": ACTIVATION_DIR / "execution_authorization.json",
        "activation": ACTIVATION_DIR / "activation_manifest.json",
    }
    if receipt_bindings is None:
        package_path, package_bytes, package = _fixed_json_file(
            expected_paths["package"],
            label="R70 package manifest",
            expected_file_sha256=PACKAGE_MANIFEST_FILE_SHA256,
            expected_canonical_sha256=PACKAGE_MANIFEST_CANONICAL_SHA256,
        )
        job_path, job_bytes, job = _fixed_json_file(
            expected_paths["job"], label="R70 job"
        )
        authorization_path, authorization_bytes, authorization = _fixed_json_file(
            expected_paths["authorization"],
            label="R70 execution authorization",
            expected_file_sha256=AUTHORIZATION_FILE_SHA256,
            expected_canonical_sha256=AUTHORIZATION_CANONICAL_SHA256,
        )
        activation_path, activation_bytes, activation = _fixed_json_file(
            expected_paths["activation"],
            label="R70 activation manifest",
            expected_file_sha256=ACTIVATION_FILE_SHA256,
            expected_canonical_sha256=ACTIVATION_CANONICAL_SHA256,
        )
    else:
        package_path, package_bytes, package = _safe_repo_binding(
            _mapping(receipt_bindings.get("package_manifest"), label="package binding"),
            label="R70 package manifest",
        )
        job_path, job_bytes, job = _safe_repo_binding(
            _mapping(receipt_bindings.get("job"), label="job binding"),
            label="R70 job",
        )
        authorization_path, authorization_bytes, authorization = _safe_repo_binding(
            _mapping(receipt_bindings.get("authorization"), label="authorization binding"),
            label="R70 execution authorization",
        )
        activation_path, activation_bytes, activation = _safe_repo_binding(
            _mapping(receipt_bindings.get("activation_manifest"), label="activation binding"),
            label="R70 activation manifest",
        )
        observed_paths = {
            "package": package_path,
            "job": job_path,
            "authorization": authorization_path,
            "activation": activation_path,
        }
        if any(
            observed_paths[key].resolve() != path.resolve()
            for key, path in expected_paths.items()
        ):
            raise AdapterInputError("retrieval receipt authority paths drifted")
    if (
        hashlib.sha256(package_bytes).hexdigest() != PACKAGE_MANIFEST_FILE_SHA256
        or package.get("sha256") != PACKAGE_MANIFEST_CANONICAL_SHA256
        or package.get("package_id") != PACKAGE_ID
        or package.get("campaign_id") != CAMPAIGN_ID
        or hashlib.sha256(authorization_bytes).hexdigest()
        != AUTHORIZATION_FILE_SHA256
        or authorization.get("sha256") != AUTHORIZATION_CANONICAL_SHA256
        or hashlib.sha256(activation_bytes).hexdigest() != ACTIVATION_FILE_SHA256
        or activation.get("sha256") != ACTIVATION_CANONICAL_SHA256
    ):
        raise AdapterInputError("R70 sealed authority identity drifted")
    if (
        job.get("execution_id") != execution_id
        or job.get("package_id") != PACKAGE_ID
        or job.get("campaign_id") != CAMPAIGN_ID
        or job.get("regime_id") != regime
        or job.get("nph") != NPH_BY_REGIME[regime]
        or job.get("route_id") != "append_singleton"
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("execution_entrypoint") != "run_append_adapt"
        or job.get("horizon") != {"source": 50, "target": 70}
        or _mapping(job.get("source_archive"), label="job source archive").get("sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise AdapterInputError("R70 singleton job contract drifted")
    _source_path, _source_bytes, source_protocol = _safe_repo_binding(
        _mapping(job.get("source_protocol"), label="source protocol binding"),
        label="source R50 protocol",
    )
    if _mapping(
        source_protocol.get("source_locks"), label="source protocol locks"
    ).get("ed_cutoff_reference_sha256") != ED_REFERENCE_SHA256:
        raise AdapterInputError("source protocol ED reference lock drifted")
    return {
        "package": (package_path, package_bytes, package),
        "job": (job_path, job_bytes, job),
        "authorization": (authorization_path, authorization_bytes, authorization),
        "activation": (activation_path, activation_bytes, activation),
    }


def _validate_embedded_payload(
    *,
    archive: Path,
    regime: str,
    authority: Mapping[str, Any],
    captured: Mapping[str, bytes],
    observed: Mapping[str, Mapping[str, Any]],
    full_worker_inventory: bool,
    source_record: Mapping[str, Any],
    receipt_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution_id = _execution_id(regime)
    package_path, package_bytes, package = authority["package"]
    job_path, job_bytes, job = authority["job"]
    authorization_path, authorization_bytes, _authorization = authority["authorization"]
    activation_path, activation_bytes, _activation = authority["activation"]
    if (
        captured["authority/job.json"] != job_bytes
        or captured["authority/execution_authorization.json"] != authorization_bytes
        or captured["authority/activation_manifest.json"] != activation_bytes
    ):
        raise AdapterInputError("archived authority bytes drifted")

    attempt_bytes = captured["worker_attempt_receipt.json"]
    attempt = _load_json_bytes(attempt_bytes, label="worker attempt receipt")
    attempt_sha = verify_self_digest(attempt, label="worker attempt receipt")
    declared_worker = _worker_inventory(attempt.get("worker_files"))
    observed_worker = {
        name.removeprefix("worker_outputs/"): dict(row)
        for name, row in observed.items()
        if name.startswith("worker_outputs/")
    }
    if full_worker_inventory:
        inventory_matches = declared_worker == observed_worker
    else:
        retained_paths = set(REQUIRED_WORKER_PATHS).difference(GIANT_WORKER_PATHS)
        inventory_matches = (
            set(observed_worker) == retained_paths
            and all(declared_worker.get(path) == row for path, row in observed_worker.items())
            and set(declared_worker).difference(observed_worker) == GIANT_WORKER_PATHS
        )
    if (
        not inventory_matches
        or attempt.get("schema")
        != "paper_i_append_adapt_stationary_core_r70_worker_attempt_v1"
        or attempt.get("execution_id") != execution_id
        or attempt.get("cluster_id") != CLUSTER_ID
        or attempt.get("proc_id") != PROC_BY_REGIME[regime]
        or attempt.get("attempt_ordinal") != 1
        or attempt.get("worker_exit_status") != 0
        or attempt.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or attempt.get("job_file_sha256") != hashlib.sha256(job_bytes).hexdigest()
        or attempt.get("authorization_file_sha256") != AUTHORIZATION_FILE_SHA256
        or attempt.get("activation_manifest_file_sha256") != ACTIVATION_FILE_SHA256
        or attempt.get("image_sha256") != IMAGE_SHA256
    ):
        raise AdapterInputError("worker attempt receipt/inventory drifted")

    worker_bytes = captured["worker_outputs/worker_receipt.json"]
    worker = _load_json_bytes(worker_bytes, label="worker receipt")
    worker_sha = verify_self_digest(worker, label="worker receipt")
    artifacts = _artifact_inventory(worker.get("artifacts"))
    checkpoint_bytes = captured["worker_outputs/payload/checkpoint.json"]
    summary_bytes = captured["worker_outputs/payload/summary.json"]
    execution_manifest_bytes = captured[
        "worker_outputs/payload/execution_manifest.json"
    ]
    checkpoint = _load_json_bytes(checkpoint_bytes, label="R70 checkpoint")
    checkpoint_sha = verify_self_digest(checkpoint, label="R70 checkpoint")
    summary = _load_json_bytes(summary_bytes, label="R70 summary")
    execution_manifest = _load_json_bytes(
        execution_manifest_bytes, label="R70 execution manifest"
    )
    execution_manifest_sha = verify_self_digest(
        execution_manifest, label="R70 execution manifest"
    )
    for role, payload in (
        ("checkpoint", checkpoint_bytes),
        ("summary", summary_bytes),
        ("execution_manifest", execution_manifest_bytes),
    ):
        artifact = artifacts[role]
        if (
            artifact.get("sha256") != hashlib.sha256(payload).hexdigest()
            or artifact.get("size_bytes") != len(payload)
        ):
            raise AdapterInputError(f"worker {role} artifact binding drifted")
    if receipt_validation is not None:
        receipt_attempt = _mapping(
            receipt_validation.get("attempt_receipt"), label="receipt attempt binding"
        )
        receipt_outputs = _mapping(
            receipt_validation.get("worker_outputs"), label="receipt worker outputs"
        )
        receipt_worker = _mapping(
            receipt_outputs.get("worker_receipt"), label="receipt worker binding"
        )
        receipt_summary = _mapping(
            receipt_outputs.get("summary"), label="receipt summary binding"
        )
        receipt_manifest = _mapping(
            receipt_outputs.get("execution_manifest"),
            label="receipt execution-manifest binding",
        )
        if (
            receipt_attempt.get("canonical_sha256") != attempt_sha
            or receipt_attempt.get("file_sha256")
            != hashlib.sha256(attempt_bytes).hexdigest()
            or receipt_attempt.get("worker_exit_status") != 0
            or receipt_worker.get("canonical_sha256") != worker_sha
            or receipt_worker.get("file_sha256")
            != hashlib.sha256(worker_bytes).hexdigest()
            or receipt_summary.get("file_sha256")
            != hashlib.sha256(summary_bytes).hexdigest()
            or receipt_summary.get("controller_rounds_completed") != 70
            or receipt_summary.get("stop_reason") != "maximum_controller_rounds"
            or receipt_manifest.get("canonical_sha256") != execution_manifest_sha
            or receipt_manifest.get("file_sha256")
            != hashlib.sha256(execution_manifest_bytes).hexdigest()
            or receipt_manifest.get("target_horizon") != 70
            or receipt_manifest.get("fresh_start") is not True
        ):
            raise AdapterInputError("retrieval receipt compact-payload binding drifted")
    if (
        worker.get("schema")
        != "paper_i_append_adapt_stationary_core_r70_worker_receipt_v1"
        or worker.get("status") != "passed"
        or worker.get("execution_id") != execution_id
        or worker.get("package_id") != PACKAGE_ID
        or worker.get("campaign_id") != CAMPAIGN_ID
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("derived_protocol_sha256") != job.get("derived_protocol_sha256")
        or worker.get("fresh_start") is not True
        or worker.get("resume_claimed") is not False
        or execution_manifest.get("schema")
        != "paper_i_append_adapt_stationary_core_r70_execution_manifest_v1"
        or execution_manifest.get("execution_id") != execution_id
        or execution_manifest.get("target_horizon") != 70
        or execution_manifest.get("fresh_start") is not True
        or execution_manifest.get("resume_claimed") is not False
        or checkpoint.get("schema")
        != "paper_i_append_adapt_reconstruction_checkpoint_v1"
        or checkpoint.get("execution_id") != execution_id
        or checkpoint.get("protocol_sha256") != job.get("derived_protocol_sha256")
        or checkpoint.get("controller_rounds_completed") != 70
        or checkpoint.get("fresh_start_execution") is not True
        or checkpoint.get("resume_claimed") is not False
        or checkpoint.get("source_checkpoint_consumed") is not False
        or checkpoint.get("source_result_consumed") is not False
        or summary.get("schema") != "paper_i_append_run_summary_v1"
        or summary.get("protocol_sha256") != job.get("derived_protocol_sha256")
        or summary.get("protocol_horizon") != 70
        or summary.get("controller_rounds_completed") != 70
        or summary.get("stop_reason") != "maximum_controller_rounds"
        or summary.get("candidate_representation") != "single_pauli_word_v1"
    ):
        raise AdapterInputError("R70 checkpoint/summary horizon closure drifted")
    return {
        "regime": regime,
        "job": job,
        "summary": summary,
        "checkpoint": checkpoint,
        "source": {
            **copy.deepcopy(dict(source_record)),
            "job": {
                "path": str(job_path.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(job_bytes).hexdigest(),
                "canonical_sha256": job.get("sha256"),
            },
            "package_manifest": {
                "path": str(package_path.relative_to(REPO_ROOT)),
                "sha256": hashlib.sha256(package_bytes).hexdigest(),
                "canonical_sha256": package.get("sha256"),
            },
            "authorization": {"path": str(authorization_path.relative_to(REPO_ROOT))},
            "activation_manifest": {"path": str(activation_path.relative_to(REPO_ROOT))},
            "worker_attempt_receipt_sha256": attempt_sha,
            "worker_receipt_sha256": worker_sha,
            "checkpoint_file_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "checkpoint_canonical_sha256": checkpoint_sha,
            "summary_file_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "execution_manifest_canonical_sha256": execution_manifest_sha,
            "retained_worker_member_hash_size_closure": True,
            "full_worker_inventory_locally_streamed": full_worker_inventory,
        },
    }


def _validate_receipt_and_archive(
    *, receipt_path: Path, archive_path: Path
) -> dict[str, Any]:
    receipt_file = _plain_file(receipt_path, label="retrieval receipt")
    archive = _plain_file(archive_path, label="retrieved archive")
    receipt, receipt_bytes = _load_json_file(receipt_file, label="retrieval receipt")
    receipt_sha = verify_self_digest(receipt, label="retrieval receipt")
    execution = _mapping(receipt.get("execution"), label="receipt execution")
    retrieval = _mapping(receipt.get("retrieval"), label="receipt retrieval")
    bindings = _mapping(receipt.get("bindings"), label="receipt bindings")
    scope = _mapping(receipt.get("receipt_scope"), label="receipt scope")
    validation = _mapping(
        receipt.get("archive_validation"), label="receipt archive validation"
    )
    member_validation_keys = (
        "gzip_and_full_tar_scan_passed",
        "safe_unique_regular_only_member_closure_passed",
        "worker_inventory_hash_size_closure_passed",
        "authority_byte_identity_passed",
        "worker_declared_fresh70_crosslink_checks_passed",
    )
    if (
        receipt.get("schema") != RETRIEVAL_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("retrieval_classification")
        != "remote_local_identity_matched_authority_and_inventory_closed"
        or receipt.get("paper_evidence_adopted") is not False
        or scope.get("transport_integrity_authenticated") is not True
        or scope.get("sealed_authority_provenance_authenticated") is not True
        or scope.get("worker_inventory_authenticated") is not True
        or scope.get("worker_exit_zero_declared") is not True
        or any(validation.get(key) is not True for key in member_validation_keys)
        or validation.get("member_count") != 12
        or validation.get("worker_file_count") != 8
    ):
        raise AdapterInputError("retrieval receipt authority/state closure failed")
    archive_sha = _sha256_file(archive)
    archive_size = archive.stat().st_size
    regime = _validate_archive_identity(
        execution=execution,
        retrieval=retrieval,
        archive_name=archive.name,
        archive_sha256=archive_sha,
        archive_size_bytes=archive_size,
    )
    authority = _local_authority(
        execution_id=_execution_id(regime),
        regime=regime,
        receipt_bindings=bindings,
    )
    captured, observed = _stream_compact_members(archive)
    return _validate_embedded_payload(
        archive=archive,
        regime=regime,
        authority=authority,
        captured=captured,
        observed=observed,
        full_worker_inventory=True,
        receipt_validation=validation,
        source_record={
            "admission_mode": SOURCE_MODE_RECEIPT_FULL,
            "retrieval_receipt": {
                "path": str(receipt_file),
                "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "canonical_sha256": receipt_sha,
            },
            "archive": {
                "path": str(archive),
                "sha256": archive_sha,
                "size_bytes": archive_size,
            },
            "transport_authentication": {
                "remote_local_full_archive_identity_authenticated": True,
                "full_archive_locally_streamed": True,
                "retrieval_receipt_present": True,
            },
            "limitations": [],
        },
    )


def _embedded_execution(captured: Mapping[str, bytes]) -> tuple[str, str]:
    job = _load_json_bytes(captured["authority/job.json"], label="embedded R70 job")
    execution_id = str(job.get("execution_id", ""))
    return execution_id, _regime_from_execution_id(execution_id)


def _validate_direct_full_archive(*, archive_path: Path) -> dict[str, Any]:
    archive = _plain_file(archive_path, label="direct full archive")
    captured, observed = _stream_compact_members(archive)
    execution_id, regime = _embedded_execution(captured)
    expected_name = (
        f"{execution_id}__cluster_{CLUSTER_ID}__proc_"
        f"{PROC_BY_REGIME[regime]}.tar.gz"
    )
    if archive.name != expected_name:
        raise AdapterInputError("direct full archive filename/attempt drifted")
    authority = _local_authority(execution_id=execution_id, regime=regime)
    archive_sha = _sha256_file(archive)
    return _validate_embedded_payload(
        archive=archive,
        regime=regime,
        authority=authority,
        captured=captured,
        observed=observed,
        full_worker_inventory=True,
        source_record={
            "admission_mode": SOURCE_MODE_DIRECT_FULL,
            "archive": {
                "path": str(archive),
                "sha256": archive_sha,
                "size_bytes": archive.stat().st_size,
            },
            "transport_authentication": {
                "remote_local_full_archive_identity_authenticated": False,
                "full_archive_locally_streamed": True,
                "retrieval_receipt_present": False,
            },
            "limitations": [
                "The complete local archive, embedded worker inventory, and sealed "
                "authority were authenticated, but no retrieval receipt independently "
                "binds these local bytes to the remote archive."
            ],
        },
    )


def _validated_observed_utc(value: str) -> str:
    from datetime import datetime

    if not isinstance(value, str) or not value.endswith("Z"):
        raise AdapterInputError("remote observation UTC must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AdapterInputError("remote observation UTC is not RFC-3339") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise AdapterInputError("remote observation timestamp is not UTC")
    return value


def _validate_compact_archive(
    *,
    archive_path: Path,
    compact_sha256: str,
    remote_full_archive_sha256: str,
    remote_full_archive_size_bytes: int,
    remote_observed_utc: str,
) -> dict[str, Any]:
    archive = _plain_file(archive_path, label="compact R70 archive")
    if (
        re.fullmatch(r"[0-9a-f]{64}", compact_sha256) is None
        or _sha256_file(archive) != compact_sha256
        or re.fullmatch(r"[0-9a-f]{64}", remote_full_archive_sha256) is None
        or remote_full_archive_sha256 == compact_sha256
        or isinstance(remote_full_archive_size_bytes, bool)
        or not isinstance(remote_full_archive_size_bytes, int)
        or remote_full_archive_size_bytes <= archive.stat().st_size
    ):
        raise AdapterInputError("compact/full archive observation binding is invalid")
    observed_utc = _validated_observed_utc(remote_observed_utc)
    captured, observed = _stream_compact_members(
        archive, expected_members=COMPACT_MEMBERS
    )
    execution_id, regime = _embedded_execution(captured)
    expected_name = (
        f"{execution_id}__cluster_{CLUSTER_ID}__proc_"
        f"{PROC_BY_REGIME[regime]}.compact.tar.gz"
    )
    if archive.name != expected_name:
        raise AdapterInputError("compact archive filename/attempt drifted")
    authority = _local_authority(execution_id=execution_id, regime=regime)
    return _validate_embedded_payload(
        archive=archive,
        regime=regime,
        authority=authority,
        captured=captured,
        observed=observed,
        full_worker_inventory=False,
        source_record={
            "admission_mode": SOURCE_MODE_COMPACT,
            "compact_archive": {
                "path": str(archive),
                "sha256": compact_sha256,
                "size_bytes": archive.stat().st_size,
                "member_count": len(COMPACT_MEMBERS),
            },
            "remote_full_archive_observation": {
                "sha256": remote_full_archive_sha256,
                "size_bytes": remote_full_archive_size_bytes,
                "observed_utc": observed_utc,
                "classification": "operator_supplied_not_locally_reauthenticated",
            },
            "transport_authentication": {
                "remote_local_full_archive_identity_authenticated": False,
                "full_archive_locally_streamed": False,
                "retrieval_receipt_present": False,
                "retained_compact_member_inventory_authenticated": True,
                "unretained_worker_paths": sorted(GIANT_WORKER_PATHS),
            },
            "limitations": [
                "Only the retained compact members were locally reauthenticated. "
                "The operator-observed remote full-archive hash, size, and timestamp "
                "are recorded but the result and estimator-ledger transport bytes "
                "were not retained or locally reauthenticated."
            ],
        },
    )


def _trace(
    *, regime: str, summary: Mapping[str, Any], exact_energy: float
) -> list[dict[str, Any]]:
    history = _sequence(summary.get("accepted_history"), label=f"{regime} accepted history")
    if len(history) != 70:
        raise AdapterInputError(f"{regime}: accepted history is not 70 rounds")
    points: list[dict[str, Any]] = []
    previous: float | None = None
    for expected_round, raw in enumerate(history, start=1):
        row = _mapping(raw, label=f"{regime} history round {expected_round}")
        before = _finite(row.get("energy_before"), label=f"{regime} energy before")
        after = _finite(row.get("energy_after"), label=f"{regime} energy after")
        if row.get("controller_round") != expected_round:
            raise AdapterInputError(f"{regime}: controller rounds are not contiguous")
        if previous is not None and not math.isclose(
            before, previous, rel_tol=1.0e-11, abs_tol=1.0e-11
        ):
            raise AdapterInputError(f"{regime}: accepted-energy continuity drifted")
        if expected_round == 1:
            points.append(
                {"round": 0, "energy": before, "delta_e": abs(before - exact_energy)}
            )
        points.append(
            {
                "round": expected_round,
                "energy": after,
                "delta_e": abs(after - exact_energy),
            }
        )
        previous = after
    final_energy = _finite(summary.get("final_energy"), label=f"{regime} final energy")
    if not math.isclose(final_energy, points[-1]["energy"], rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AdapterInputError(f"{regime}: final energy drifted from round 70")
    return points


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.absolute()
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise AdapterInputError("adapter output parent is unavailable or unsafe")
    encoded = canonical_json_bytes(payload) + b"\n"
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise AdapterInputError(f"stale temporary output exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _compile_costs_for_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    request = {
        "schema": COMPILE_REQUEST_SCHEMA,
        "cells": [
            {
                "execution_id": cell["job"]["execution_id"],
                "job": cell["job"],
                "checkpoint": cell["checkpoint"],
                "summary": cell["summary"],
            }
            for cell in cells
        ],
    }
    with tempfile.TemporaryDirectory(prefix="paper_i_append_r70_compile_") as raw:
        root = Path(raw)
        request_path = root / "request.json"
        response_path = root / "response.json"
        request_path.write_bytes(canonical_json_bytes(request) + b"\n")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
        environment["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--compile-request",
                str(request_path),
                "--compile-response",
                str(response_path),
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise AdapterInputError(
                "source-locked R70 prefix compilation failed:\n" + completed.stdout[-8000:]
            )
        response, _ = _load_json_file(response_path, label="R70 compile response")
    if response.get("schema") != COMPILE_RESPONSE_SCHEMA:
        raise AdapterInputError("R70 compile response schema drifted")
    rows = _sequence(response.get("cells"), label="R70 compiled cells")
    result: dict[str, Any] = {}
    for raw in rows:
        row = _mapping(raw, label="R70 compiled cell")
        execution_id = str(row.get("execution_id", ""))
        if not execution_id or execution_id in result:
            raise AdapterInputError("R70 compile response duplicates a cell")
        result[execution_id] = dict(row)
    if set(result) != {str(cell["job"]["execution_id"]) for cell in cells}:
        raise AdapterInputError("R70 compile response cell closure drifted")
    return result


def _load_mixed_source(spec: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(spec.get("mode", ""))
    if mode == SOURCE_MODE_RECEIPT_FULL:
        return _validate_receipt_and_archive(
            receipt_path=Path(str(spec["receipt"])),
            archive_path=Path(str(spec["archive"])),
        )
    if mode == SOURCE_MODE_DIRECT_FULL:
        return _validate_direct_full_archive(
            archive_path=Path(str(spec["archive"]))
        )
    if mode == SOURCE_MODE_COMPACT:
        return _validate_compact_archive(
            archive_path=Path(str(spec["archive"])),
            compact_sha256=str(spec["compact_sha256"]),
            remote_full_archive_sha256=str(spec["remote_full_archive_sha256"]),
            remote_full_archive_size_bytes=_integer(
                spec["remote_full_archive_size_bytes"],
                label="remote full archive size",
                minimum=1,
            ),
            remote_observed_utc=str(spec["remote_observed_utc"]),
        )
    raise AdapterInputError(f"unsupported R70 source admission mode: {mode!r}")


def build_adapter(
    *,
    output: Path,
    source_specs: Sequence[Mapping[str, Any]] | None = None,
    sources: Sequence[tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    if source_specs is not None and sources is not None:
        raise AdapterInputError("use source_specs or legacy sources, not both")
    if source_specs is None:
        source_specs = [
            {
                "mode": SOURCE_MODE_RECEIPT_FULL,
                "receipt": receipt,
                "archive": archive,
            }
            for receipt, archive in (sources or ())
        ]
    if len(source_specs) != len(COMPLETED_REGIMES):
        raise AdapterInputError("exactly six completed singleton sources are required")
    exact_by_regime, ed_binding = _ed_reference()
    raw_cells = [_load_mixed_source(spec) for spec in source_specs]
    by_regime = {str(cell["regime"]): cell for cell in raw_cells}
    if len(by_regime) != len(raw_cells) or set(by_regime) != set(COMPLETED_REGIMES):
        raise AdapterInputError("completed R70 regime closure drifted")
    compiled = _compile_costs_for_cells(raw_cells)
    cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        if regime not in by_regime:
            continue
        raw = by_regime[regime]
        job = _mapping(raw["job"], label=f"{regime} job")
        summary = _mapping(raw["summary"], label=f"{regime} summary")
        points = _trace(
            regime=regime,
            summary=summary,
            exact_energy=exact_by_regime[regime],
        )
        compiled_row = _mapping(
            compiled[str(job["execution_id"])], label=f"{regime} compiled row"
        )
        endpoints: dict[str, Any] = {}
        for controller_round in (50, 70):
            key = f"round_{controller_round}"
            observation = _mapping(compiled_row.get(key), label=f"{regime} {key}")
            costs = _mapping(observation.get("costs"), label=f"{regime} {key} costs")
            if set(costs) != set(COST_FIELDS):
                raise AdapterInputError(f"{regime}: {key} cost tuple drifted")
            point = points[controller_round]
            endpoints[key] = {
                "round": controller_round,
                "energy": point["energy"],
                "delta_e": point["delta_e"],
                "checkpoint_sha256": observation.get("checkpoint_sha256"),
                "costs": {field: _integer(costs[field], label=f"{regime} {field}") for field in COST_FIELDS},
                "compile": dict(
                    _mapping(observation.get("compile"), label=f"{regime} {key} compile")
                ),
            }
        cells.append(
            {
                "regime_id": regime,
                "display_name": REGIME_DISPLAY[regime],
                "nph": NPH_BY_REGIME[regime],
                "execution_id": job["execution_id"],
                "exact_same_cutoff_energy": exact_by_regime[regime],
                "source": raw["source"],
                "points": points,
                "endpoints": endpoints,
            }
        )
    source_modes = {
        regime: str(_mapping(by_regime[regime]["source"], label="source").get("admission_mode"))
        for regime in COMPLETED_REGIMES
    }
    limitations: list[str] = []
    for regime in COMPLETED_REGIMES:
        source = _mapping(by_regime[regime]["source"], label=f"{regime} source")
        for raw in _sequence(source.get("limitations", []), label="source limitations"):
            limitation = f"{REGIME_DISPLAY[regime]}: {str(raw)}"
            if limitation not in limitations:
                limitations.append(limitation)
    adapter = digested(
        {
            "schema": ADAPTER_SCHEMA,
            "status": "passed",
            "classification": "diagnostic_not_paper_evidence",
            "package_id": PACKAGE_ID,
            "cluster_id": CLUSTER_ID,
            "regime_order": list(REGIME_ORDER),
            "completed_regimes": list(COMPLETED_REGIMES),
            "pending_regimes": list(PENDING_REGIMES),
            "source_authentication_summary": {
                "by_regime": source_modes,
                "all_retained_members_bound_to_embedded_worker_inventories": True,
                "all_embedded_authority_bytes_match_local_sealed_authority": True,
                "all_full_archives_remote_local_identity_authenticated": all(
                    mode == SOURCE_MODE_RECEIPT_FULL for mode in source_modes.values()
                ),
                "paper_evidence_adopted": False,
            },
            "limitations": limitations,
            "same_cutoff_reference": ed_binding,
            "cost_policy": {
                "tuple_fields": list(COST_FIELDS),
                "round_50": {
                    "classification": "canonical_paper_comparable",
                    "controller_round": 50,
                    "source": "authenticated_signed_prefix_shared_qiskit_recompile_v1",
                    "compile_convention": "table_i_basis_gate_transpile_v1",
                },
                "round_70": {
                    "classification": "diagnostic_extension",
                    "controller_round": 70,
                    "source": "authenticated_signed_prefix_shared_qiskit_recompile_v1",
                    "compile_convention": "table_i_basis_gate_transpile_v1",
                    "serialized_terminal_cross_check_required": True,
                },
            },
            "cells": cells,
        }
    )
    if output.exists() or output.is_symlink():
        existing, existing_bytes = _load_json_file(output, label="existing R70 adapter")
        verify_self_digest(existing, label="existing R70 adapter")
        if canonical_json_bytes(existing) != canonical_json_bytes(adapter):
            raise AdapterInputError("refusing to replace a different R70 adapter")
        return {
            "status": "already_current",
            "output": str(output),
            "sha256": adapter["sha256"],
            "file_sha256": hashlib.sha256(existing_bytes).hexdigest(),
        }
    _write_json_atomic(output, adapter)
    return {
        "status": "written",
        "output": str(output),
        "sha256": adapter["sha256"],
        "file_sha256": _sha256_file(output),
    }


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterInputError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _safe_extract_source(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as bundle:
        names: set[str] = set()
        for member in bundle:
            name = _safe_member_name(member.name)
            if name in names or not member.isfile() or member.issym() or member.islnk():
                raise AdapterInputError(f"unsafe source archive member: {name}")
            names.add(name)
        bundle.extractall(destination, filter="data")


def _prefix_from_checkpoint(
    *, job: Mapping[str, Any], checkpoint_document: Mapping[str, Any], protocol: Any, problem: Any, controller_round: int
) -> Any:
    """Reconstruct one signed Append prefix inside the locked source tree."""

    import numpy as np

    from pipelines.reporting.paper_i_run_summary import (
        PaperIAlgorithmicWork,
        PaperIPrefixCompileInput,
        PaperIPrefixOperator,
        PaperIPrefixPauliTerm,
        PaperIReferenceState,
        PaperIWorkComponents,
    )
    from pipelines.static_adapt.estimator_call_ledger import projective_state_fingerprint
    from pipelines.static_adapt.ra_adapt.append import _validate_resolved_append_protocol
    from pipelines.static_adapt.ra_adapt.replay_evidence import (
        _prefix_replay_identity,
        _verify_signed,
    )

    replay = _mapping(
        checkpoint_document.get("controller_replay_evidence"), label="controller replay evidence"
    )
    prefixes = _sequence(replay.get("signed_controller_round_prefixes"), label="signed prefixes")
    if len(prefixes) != 70:
        raise AdapterInputError("signed prefix closure is not 70 rounds")
    selected = _mapping(prefixes[controller_round - 1], label="selected signed prefix")
    verified = _verify_signed(selected, name="selected Append controller prefix")
    active = _verify_signed(
        verified.get("active_prefix_checkpoint"),
        name="selected Append active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    route_identity = _mapping(verified.get("route_identity"), label="selected route identity")
    if (
        verified.get("schema") != "paper_i_signed_controller_round_prefix_v1"
        or verified.get("method_family") != "append_adapt"
        or verified.get("controller_round") != controller_round
        or verified.get("protocol_sha256") != protocol.sha256
        or verified.get("source_checkpoint_sha256") != active.get("checkpoint_sha256")
        or route_identity
        != {"selector_identity": protocol.selector_identity, "selector_scope": protocol.selector_scope}
        or active.get("schema") != "paper_i_signed_append_active_prefix_checkpoint_v1"
        or active.get("controller_round") != controller_round
        or active.get("protocol_sha256") != protocol.sha256
    ):
        raise AdapterInputError(f"{job['execution_id']}: selected prefix identity drifted")
    replay_identity = _prefix_replay_identity(
        method_family="append_adapt",
        problem_request_sha256=protocol.problem.problem_request_sha256,
        route_identity=route_identity,
        controller_round=controller_round,
        operator_labels=active["accepted_operator_labels"],
        logical_parameters=active["logical_parameters"],
        runtime_parameters=active["runtime_parameters"],
        state_fingerprint=active["projective_state_fingerprint"],
        accepted_energy=float(active["accepted_energy"]),
    )
    if replay_identity != verified.get("prefix_replay_identity_sha256"):
        raise AdapterInputError(f"{job['execution_id']}: prefix replay identity drifted")

    _request, _parent, executable_inventory, _lineage = _validate_resolved_append_protocol(
        problem, protocol
    )
    labels = tuple(str(value) for value in active["accepted_operator_labels"])
    identities = tuple(str(value) for value in active["accepted_generator_identities"])
    logical = tuple(_finite(value, label="logical parameter") for value in active["logical_parameters"])
    runtime = tuple(_finite(value, label="runtime parameter") for value in active["runtime_parameters"])
    if len(labels) != controller_round or len(identities) != controller_round or len(logical) != controller_round:
        raise AdapterInputError("selected Append lineage length drifted")
    candidates = {str(candidate.label): candidate for candidate in executable_inventory.candidates}
    if len(candidates) != len(executable_inventory.candidates):
        raise AdapterInputError("Append executable pool duplicates labels")
    operators: list[Any] = []
    runtime_start = 0
    for logical_index, (label, identity) in enumerate(zip(labels, identities, strict=True)):
        candidate = candidates.get(label)
        if candidate is None or str(candidate.generator_identity) != identity:
            raise AdapterInputError("selected generator left the protocol-locked pool")
        terms = tuple(
            PaperIPrefixPauliTerm(
                pauli_exyz=str(term["pauli_exyz"]),
                coefficient_real=_finite(term.get("coeff_re"), label="Pauli coefficient real"),
                coefficient_imaginary=_finite(term.get("coeff_im"), label="Pauli coefficient imaginary"),
                qubit_count=_integer(term.get("nq"), label="Pauli qubit count", minimum=1),
            )
            for term in candidate.serialized_terms_exyz
        )
        if not terms:
            raise AdapterInputError("selected Append candidate has no terms")
        operators.append(
            PaperIPrefixOperator(
                candidate_label=label,
                logical_index=logical_index,
                runtime_start=runtime_start,
                runtime_count=len(terms),
                execution_mode=str(candidate.execution_mode),
                runtime_terms=terms,
            )
        )
        runtime_start += len(terms)
    if runtime_start != len(runtime):
        raise AdapterInputError("Append runtime-parameter partition drifted")
    reference_array = np.asarray(problem.reference_state.build_state(), dtype=complex).reshape(-1)
    reference_array = reference_array / float(np.linalg.norm(reference_array))
    reference = PaperIReferenceState(
        amplitudes_real=tuple(float(value.real) for value in reference_array),
        amplitudes_imaginary=tuple(float(value.imag) for value in reference_array),
        qubit_count=int(problem.layout.total_qubits),
        source_label=str(problem.reference_state.source_label),
        state_fingerprint=projective_state_fingerprint(reference_array),
    )
    estimator_prefix = _mapping(active.get("estimator_prefix"), label="estimator prefix")
    executed = _mapping(estimator_prefix.get("cumulative_executed_queries"), label="executed queries")
    component_raw = _mapping(executed.get("components"), label="work components")
    components = PaperIWorkComponents(
        n_h_outer=_integer(component_raw.get("N_H_outer"), label="N_H_outer"),
        n_h_refit=_integer(component_raw.get("N_H_refit"), label="N_H_refit"),
        n_grad=_integer(component_raw.get("N_grad"), label="N_grad"),
        n_metric=_integer(component_raw.get("N_metric"), label="N_metric"),
    )
    s_alg = _integer(executed.get("S_alg"), label="S_alg")
    if components.s_alg != s_alg:
        raise AdapterInputError("selected Append S_alg drifted")
    route_contract = _mapping(protocol.route_contract, label="Append route contract")
    return PaperIPrefixCompileInput(
        source_method="append_adapt",
        controller_round=controller_round,
        active_ansatz_depth=len(labels),
        ordered_operator_labels=labels,
        operators=tuple(operators),
        logical_parameters=logical,
        runtime_parameters=runtime,
        reference_state=reference,
        checkpoint_sha256=str(active["checkpoint_sha256"]),
        projective_state_fingerprint=str(active["projective_state_fingerprint"]),
        problem_request_sha256=str(protocol.problem.problem_request_sha256),
        route_profile=str(route_contract.get("route_profile", "")),
        route_contract_sha256=str(route_contract.get("sha256", "")),
        algorithmic_work=PaperIAlgorithmicWork(components=components, s_alg=s_alg),
    )


def _cost_from_payload(prefix: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    costs = {
        "N2q": _integer(payload.get("compiled_count_2q_total"), label="compiled N2q"),
        "D2q": _integer(payload.get("compiled_depth_2q_total"), label="compiled D2q"),
        "Dc": _integer(payload.get("compiled_depth_total"), label="compiled Dc"),
        "W1q": _integer(
            payload.get("qiskit_pretranspile_pauli_1q_work_total"), label="compiled W1q"
        ),
        "S_alg": _integer(prefix.algorithmic_work.s_alg, label="compiled-prefix S_alg"),
    }
    if payload.get("compile_convention") != "table_i_basis_gate_transpile_v1":
        raise AdapterInputError("Paper-I Qiskit compile convention drifted")
    return {
        "checkpoint_sha256": str(prefix.checkpoint_sha256),
        "costs": costs,
        "compile": {
            "compile_convention": payload.get("compile_convention"),
            "qiskit_version": payload.get("qiskit_version"),
            "compiled_basis_gates": payload.get("compiled_basis_gates"),
            "qiskit_transpile_optimization_level": payload.get(
                "qiskit_transpile_optimization_level"
            ),
            "qiskit_transpile_seed": payload.get("qiskit_transpile_seed"),
            "compiled_circuit_scope": payload.get("compiled_circuit_scope"),
            "generator_coefficients_sha256": payload.get("generator_coefficients_sha256"),
            "source": "PaperIPrefixCompileInput_to_shared_locked_compiler_v1",
        },
    }


def _terminal_cross_check(
    *, summary: Mapping[str, Any], observation: Mapping[str, Any]
) -> None:
    resources = _mapping(summary.get("resources"), label="summary resources")
    terminal = _mapping(
        resources.get("terminal_compiled_resources"), label="serialized terminal resources"
    )
    costs = _mapping(observation.get("costs"), label="compiled terminal costs")
    expected = {
        "N2q": terminal.get("compiled_count_2q_total"),
        "D2q": terminal.get("compiled_depth_2q_total"),
        "Dc": terminal.get("compiled_depth_total"),
        "W1q": terminal.get("qiskit_pretranspile_pauli_1q_work_total"),
        "S_alg": _mapping(summary.get("estimator_accounting"), label="terminal accounting").get("S_alg"),
    }
    if dict(costs) != expected:
        raise AdapterInputError("round-70 shared Qiskit recompile cross-check failed")


def _compile_prefix_qiskit_payload_locked(prefix: Any) -> dict[str, Any]:
    """Cross the same locked Table-I compiler boundary as Paper-I summaries.

    The source archive predates the later public payload-returning wrapper in
    ``paper_i_run_summary``.  Its private compiler calls this exact lower-level
    routine but projects away W1q and compiler metadata.  Calling the same
    routine here preserves those fields without consulting ambient source.
    """

    import numpy as np

    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        TABLE_I_COMPILED_BASIS_GATES,
        TABLE_I_QISKIT_COMPILE_CONVENTION,
        TableIQiskitCompileConfig,
        compile_table_i_ansatz_terms,
    )
    from src.quantum.pauli_polynomial_class import PauliPolynomial
    from src.quantum.qubitization_module import PauliTerm
    from src.quantum.vqe_latex_python_pairs import AnsatzTerm

    if TABLE_I_QISKIT_COMPILE_CONVENTION != "table_i_basis_gate_transpile_v1":
        raise AdapterInputError("source-locked Table-I compile convention drifted")
    operators = []
    for operator in prefix.operators:
        polynomial = PauliPolynomial("JW")
        for term in operator.runtime_terms:
            polynomial.add_term(
                PauliTerm(
                    term.qubit_count,
                    ps=term.pauli_exyz,
                    pc=complex(
                        term.coefficient_real,
                        term.coefficient_imaginary,
                    ),
                )
            )
        operators.append(
            AnsatzTerm(
                label=operator.candidate_label,
                polynomial=polynomial,
                execution_mode=operator.execution_mode,
            )
        )
    reference = np.asarray(
        prefix.reference_state.amplitudes_real,
        dtype=float,
    ).astype(complex)
    reference += 1.0j * np.asarray(
        prefix.reference_state.amplitudes_imaginary,
        dtype=float,
    )
    payload = compile_table_i_ansatz_terms(
        ops=tuple(operators),
        num_qubits=prefix.reference_state.qubit_count,
        reference_state=reference,
        source_kind="canonical_paper_i_accepted_prefix",
        config=TableIQiskitCompileConfig(
            basis_gates=TABLE_I_COMPILED_BASIS_GATES,
            optimization_level=0,
            seed_transpiler=7,
            structure_theta_value=1.0,
            include_reference_state=True,
            compile_convention="table_i_basis_gate_transpile_v1",
            coefficient_tolerance=1.0e-12,
            grouped_exact_max_active_qubits=5,
        ),
    )
    if (
        tuple(payload.get("compiled_basis_gates", ()))
        != TABLE_I_COMPILED_BASIS_GATES
        or int(payload.get("qiskit_transpile_optimization_level", -1)) != 0
        or int(payload.get("qiskit_transpile_seed", -1)) != 7
        or payload.get("compiled_circuit_scope")
        != "ansatz_circuit_including_reference_state"
    ):
        raise AdapterInputError("source-locked Paper-I compiler settings drifted")
    return dict(payload)


def _compile_request_mode(request_path: Path, response_path: Path) -> int:
    request, _ = _load_json_file(request_path, label="R70 compile request")
    if request.get("schema") != COMPILE_REQUEST_SCHEMA:
        raise AdapterInputError("R70 compile request schema drifted")
    sys.dont_write_bytecode = True
    package_text = str(PACKAGE_DIR)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    contract = _load_module(PACKAGE_DIR / "package_contract.py", "package_contract")
    derived = _load_module(PACKAGE_DIR / "derived_protocol.py", "derived_protocol")
    validation = contract.validate_package(full_archive_scan=True)
    if validation.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise AdapterInputError("source-locked package validation drifted")
    archive = PACKAGE_DIR / contract.SOURCE_ARCHIVE_NAME
    if _sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise AdapterInputError("source-locked archive hash drifted")
    with tempfile.TemporaryDirectory(prefix="paper_i_append_r70_source_") as raw:
        source_root = Path(raw) / "source_locked"
        source_root.mkdir()
        _safe_extract_source(archive, source_root)
        derived.activate_source_root(source_root)
        output_rows: list[dict[str, Any]] = []
        for raw_cell in _sequence(request.get("cells"), label="compile request cells"):
            cell = _mapping(raw_cell, label="compile request cell")
            job = _mapping(cell.get("job"), label="compile job")
            checkpoint = _mapping(cell.get("checkpoint"), label="compile checkpoint")
            summary = _mapping(cell.get("summary"), label="compile summary")
            protocol, problem, audit = derived.build_derived_protocol(
                job=job,
                source_root=source_root,
                validate_entire_bundle=False,
            )
            if (
                protocol.sha256 != job.get("derived_protocol_sha256")
                or audit.get("normalized_non_horizon_settings_match") is not True
            ):
                raise AdapterInputError("derived R70 protocol reconstruction drifted")
            observations: dict[str, Any] = {}
            for controller_round in (50, 70):
                prefix = _prefix_from_checkpoint(
                    job=job,
                    checkpoint_document=checkpoint,
                    protocol=protocol,
                    problem=problem,
                    controller_round=controller_round,
                )
                payload = _compile_prefix_qiskit_payload_locked(prefix)
                observations[f"round_{controller_round}"] = _cost_from_payload(prefix, payload)
            _terminal_cross_check(summary=summary, observation=observations["round_70"])
            output_rows.append(
                {"execution_id": str(job["execution_id"]), **observations}
            )
    response = {"schema": COMPILE_RESPONSE_SCHEMA, "cells": output_rows}
    _write_json_atomic(response_path, response)
    return 0


def _parse_source(value: Sequence[str]) -> tuple[Path, Path]:
    if len(value) != 2:
        raise argparse.ArgumentTypeError("--source requires RECEIPT ARCHIVE")
    return Path(value[0]), Path(value[1])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("RETRIEVAL_RECEIPT", "ARCHIVE"),
        help="Receipt-authenticated full archive; may be mixed with other modes.",
    )
    parser.add_argument(
        "--direct-full-archive",
        type=Path,
        action="append",
        help=(
            "Full local archive without a retrieval receipt; it is fully streamed "
            "and bound to embedded/local sealed authority, while remote transport "
            "identity remains explicitly unauthenticated."
        ),
    )
    parser.add_argument(
        "--compact-source",
        nargs=5,
        action="append",
        metavar=(
            "COMPACT_ARCHIVE",
            "COMPACT_SHA256",
            "REMOTE_FULL_SHA256",
            "REMOTE_FULL_SIZE",
            "REMOTE_OBSERVED_UTC",
        ),
        help=(
            "Compact diagnostic archive and the operator-observed remote full-"
            "archive identity; giant-member transport is not locally reauthenticated."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compile-response", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.compile_request is not None or args.compile_response is not None:
            if args.compile_request is None or args.compile_response is None:
                raise AdapterInputError("private compile mode requires both paths")
            return _compile_request_mode(args.compile_request, args.compile_response)
        if args.output is None:
            raise AdapterInputError("--output is required")
        source_specs: list[dict[str, Any]] = []
        source_specs.extend(
            {
                "mode": SOURCE_MODE_RECEIPT_FULL,
                "receipt": Path(row[0]),
                "archive": Path(row[1]),
            }
            for row in (args.source or ())
        )
        source_specs.extend(
            {"mode": SOURCE_MODE_DIRECT_FULL, "archive": path}
            for path in (args.direct_full_archive or ())
        )
        for row in args.compact_source or ():
            try:
                remote_size = int(row[3])
            except ValueError as exc:
                raise AdapterInputError(
                    "compact remote full archive size must be an integer"
                ) from exc
            source_specs.append(
                {
                    "mode": SOURCE_MODE_COMPACT,
                    "archive": Path(row[0]),
                    "compact_sha256": row[1],
                    "remote_full_archive_sha256": row[2],
                    "remote_full_archive_size_bytes": remote_size,
                    "remote_observed_utc": row[4],
                }
            )
        result = build_adapter(
            source_specs=source_specs,
            output=args.output,
        )
        print(canonical_json_bytes(result).decode("ascii"))
        return 0
    except (AdapterInputError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
