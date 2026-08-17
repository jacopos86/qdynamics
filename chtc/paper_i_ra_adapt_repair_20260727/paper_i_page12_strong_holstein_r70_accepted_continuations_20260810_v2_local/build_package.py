#!/usr/bin/env python3
"""Seal the local-only Page-12 strong-sector accepted-state continuations."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    ACTIVATION_SCHEMA,
    ALGORITHM_ID,
    AUTHORIZATION_SCHEMA,
    BASE_PACKAGE_MANIFEST_FILE_SHA256,
    BASE_PACKAGE_MANIFEST_SHA256,
    BASE_PACKAGE_RELATIVE,
    BASE_PROTOCOL_ROOT,
    BASE_SOURCE_ARCHIVE_SHA256,
    BASE_SOURCE_LOCKS_FILE_SHA256,
    BASE_SOURCE_LOCKS_SHA256,
    BASE_SOURCE_MANIFEST_FILE_SHA256,
    BASE_SOURCE_MANIFEST_SHA256,
    BUNDLE_ID,
    BUNDLE_MANIFEST_SCHEMA,
    CAMPAIGN_ID,
    CANDIDATE_ADAPTER_ID,
    CANDIDATE_REPRESENTATION,
    CELL_SPECS,
    CONTROLLER_AFTER_SHA256,
    CONTROLLER_BEFORE_SHA256,
    CONTROLLER_REGRESSION,
    CONTROLLER_RELATIVE_PATH,
    CONTROLLER_REPAIR_ID,
    CONTROL_FILES,
    EXECUTION_TARGET,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    RESOURCE_ENVELOPE,
    RESOURCE_WEIGHTING_SCOPE,
    RESUME_AFTER_SHA256,
    RESUME_AFTER_SIZE_BYTES,
    RESUME_BEFORE_SHA256,
    RESUME_REGRESSION,
    RESUME_RELATIVE_PATH,
    RESUME_REPAIR_ID,
    ROUTE_CONTRACT_SHA256,
    ROUTE_ID,
    RUN_CLASS,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TARGET_ROUTE_PROFILE,
    VENDORED_STREAMING_JSON_BACKEND,
    VENDORED_STREAMING_JSON_FILES,
    VENDORED_STREAMING_JSON_VERSION,
    V1_PACKAGE_MANIFEST_FILE_SHA256,
    V1_PACKAGE_MANIFEST_SHA256,
    V1_PACKAGE_RELATIVE,
    PackageContractError,
    canonical_json_bytes,
    digested,
    execution_id,
    expected_execution_ids,
    file_binding,
    load_json,
    repo_root_from_script,
    sha256_file,
    source_execution_id,
    validate_resume_archive,
    verify_self_digest,
)


REPO_ROOT = repo_root_from_script(__file__)
BASE_PACKAGE = REPO_ROOT / BASE_PACKAGE_RELATIVE
V1_PACKAGE = REPO_ROOT / V1_PACKAGE_RELATIVE
GENERATED_TARGETS = (
    "bundle",
    "source",
    "resume_inputs",
    "jobs",
    "activation",
    "execution_plan.json",
    "package_manifest.json",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_file(path: Path, *, sha256: str, size_bytes: int | None = None) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or (size_bytes is not None and path.stat().st_size != size_bytes)
        or sha256_file(path) != sha256
    ):
        raise PackageContractError(f"Source binding drifted: {path}")


def _copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink() or destination.exists():
        raise PackageContractError(f"Unsafe exact copy: {source} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise PackageContractError(f"Exact copy drifted: {destination}")


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        block = self.source.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block


def _tar_add_stream(
    archive: tarfile.TarFile,
    *,
    name: str,
    size: int,
    source: BinaryIO,
) -> None:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    archive.addfile(info, source)


def _load_self_digested_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Malformed {label} in source archive.") from exc
    if not isinstance(payload, dict):
        raise PackageContractError(f"{label} must be an object.")
    verify_self_digest(payload, label=label)
    return payload


def _build_resume_archive(
    spec: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    source_binding = spec["source_archive"]
    source_path = REPO_ROOT / str(source_binding["path"])
    _verify_file(
        source_path,
        sha256=str(source_binding["sha256"]),
        size_bytes=int(source_binding["size_bytes"]),
    )
    by_source = {
        str(row["source_member"]): row for row in spec["source_members"]
    }
    execution_member = (
        f"./runs/{source_execution_id(str(spec['regime_id']))}/"
        "execution_manifest.json"
    )
    observed: set[str] = set()
    small: dict[str, bytes] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(
                mode="w", fileobj=gz, format=tarfile.PAX_FORMAT
            ) as output:
                with tarfile.open(source_path, "r:gz") as source_archive:
                    for member in source_archive:
                        if member.name in {"./worker_receipt.json", execution_member}:
                            source = source_archive.extractfile(member)
                            if source is None or member.size > 1024 * 1024:
                                raise PackageContractError(
                                    f"Unsafe source authority member: {member.name}"
                                )
                            small[member.name] = source.read()
                            continue
                        row = by_source.get(member.name)
                        if row is None:
                            continue
                        if (
                            member.name in observed
                            or not member.isfile()
                            or member.issym()
                            or member.islnk()
                            or member.size != int(row["size_bytes"])
                        ):
                            raise PackageContractError(
                                f"Unsafe resume source member: {member.name}"
                            )
                        source = source_archive.extractfile(member)
                        if source is None:
                            raise PackageContractError(
                                f"Unreadable resume source member: {member.name}"
                            )
                        checked = _HashingReader(source)
                        _tar_add_stream(
                            output,
                            name=str(row["archive_path"]),
                            size=int(row["size_bytes"]),
                            source=checked,
                        )
                        if (
                            checked.size != int(row["size_bytes"])
                            or checked.digest.hexdigest() != row["sha256"]
                        ):
                            raise PackageContractError(
                                f"Resume source bytes drifted: {member.name}"
                            )
                        observed.add(member.name)
    if observed != set(by_source) or set(small) != {
        "./worker_receipt.json",
        execution_member,
    }:
        raise PackageContractError("Source attempt closure is incomplete.")
    worker = _load_self_digested_bytes(
        small["./worker_receipt.json"], label="source worker receipt"
    )
    execution = _load_self_digested_bytes(
        small[execution_member], label="source execution manifest"
    )
    source_id = source_execution_id(str(spec["regime_id"]))
    if (
        worker.get("status") != "passed"
        or worker.get("execution_id") != source_id
        or worker.get("controller_rounds_completed") != SOURCE_HORIZON
        or worker.get("job_spec_sha256") != spec["source_job_sha256"]
        or worker.get("execution_manifest_sha256") != execution.get("sha256")
        or execution.get("status") != "passed"
        or execution.get("execution_id") != source_id
        or execution.get("controller_rounds_completed") != SOURCE_HORIZON
        or execution.get("target_horizon") != SOURCE_HORIZON
        or execution.get("job_spec_sha256") != spec["source_job_sha256"]
        or execution.get("protocol_sha256") != spec["source_protocol_sha256"]
        or execution.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError(
            f"Source attempt is not a passed Page-12 k=50 cell: {source_id}"
        )
    return {
        "archive": dict(source_binding),
        "worker_receipt_sha256": worker["sha256"],
        "execution_manifest_sha256": execution["sha256"],
        "execution_id": source_id,
        "controller_rounds_completed": SOURCE_HORIZON,
        "status": "passed_authenticated_source_attempt",
    }


def _materialize_resume(
    spec: Mapping[str, Any], *, v1_package_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Remint v2 authority around v1's exact compact archive bytes.

    The sealed v1 package already authenticated every uncompressed member and
    parsed the complete checkpoint.  V2 deliberately does not reopen the
    original fetched attempt archives or stream-parse the multi-gigabyte
    checkpoint while sealing.  The worker still authenticates and extracts
    every compact-archive member before scientific execution.
    """

    regime = str(spec["regime_id"])
    v1_root = V1_PACKAGE / "resume_inputs"
    v1_archive_path = v1_root / f"{regime}.tar.gz"
    v1_manifest_path = v1_root / f"{regime}.manifest.json"
    v1_receipt_path = (
        v1_root / f"{regime}.checkpoint_validation.json"
    )
    archive_authority = spec["v1_resume_archive"]
    manifest_authority = spec["v1_resume_manifest"]
    receipt_authority = spec["v1_checkpoint_validation"]
    _verify_file(
        v1_archive_path,
        sha256=str(archive_authority["sha256"]),
        size_bytes=int(archive_authority["size_bytes"]),
    )
    _verify_file(
        v1_manifest_path,
        sha256=str(manifest_authority["sha256"]),
        size_bytes=int(manifest_authority["size_bytes"]),
    )
    _verify_file(
        v1_receipt_path,
        sha256=str(receipt_authority["sha256"]),
        size_bytes=int(receipt_authority["size_bytes"]),
    )
    v1_manifest = load_json(v1_manifest_path, label=f"{regime} v1 resume")
    v1_receipt = load_json(
        v1_receipt_path, label=f"{regime} v1 checkpoint validation"
    )
    if (
        verify_self_digest(v1_manifest, label=f"{regime} v1 resume")
        != manifest_authority["canonical_sha256"]
        or verify_self_digest(
            v1_receipt, label=f"{regime} v1 checkpoint validation"
        )
        != receipt_authority["canonical_sha256"]
    ):
        raise PackageContractError(f"V1 resume authority drifted: {regime}")
    v1_rows = [
        row
        for row in v1_package_manifest.get("resume_inputs", [])
        if isinstance(row, Mapping) and row.get("regime_id") == regime
    ]
    expected_v1_archive = {
        "path": f"resume_inputs/{regime}.tar.gz",
        **dict(archive_authority),
    }
    expected_v1_manifest = {
        "path": f"resume_inputs/{regime}.manifest.json",
        **dict(manifest_authority),
    }
    expected_v1_receipt = {
        "path": f"resume_inputs/{regime}.checkpoint_validation.json",
        **dict(receipt_authority),
    }
    if (
        len(v1_rows) != 1
        or v1_rows[0].get("archive") != expected_v1_archive
        or v1_rows[0].get("manifest") != expected_v1_manifest
        or v1_rows[0].get("checkpoint_validation") != expected_v1_receipt
        or v1_manifest.get("archive") != expected_v1_archive
        or v1_manifest.get("checkpoint_validation") != expected_v1_receipt
        or v1_receipt.get("archive") != expected_v1_archive
        or v1_receipt.get("members") != v1_manifest.get("members")
        or v1_manifest.get("package_id")
        != "paper_i_page12_strong_holstein_r70_accepted_continuations_20260810_v1_local"
        or v1_receipt.get("package_id")
        != "paper_i_page12_strong_holstein_r70_accepted_continuations_20260810_v1_local"
        or v1_manifest.get("resume_round") != SOURCE_HORIZON
        or v1_manifest.get("target_round") != TARGET_HORIZON
        or v1_receipt.get("status") != "passed"
    ):
        raise PackageContractError(
            f"Sealed v1 resume closure drifted: {regime}"
        )
    archive_path = PACKAGE_DIR / "resume_inputs" / f"{regime}.tar.gz"
    _copy_exact(v1_archive_path, archive_path)
    archive_binding = file_binding(archive_path, root=PACKAGE_DIR)
    if archive_binding != expected_v1_archive:
        raise PackageContractError(f"V2 archive bytes drifted: {regime}")
    members = list(v1_manifest["members"])
    inherited = {
        "package": {
            "path": V1_PACKAGE_RELATIVE.as_posix(),
            "manifest_file_sha256": V1_PACKAGE_MANIFEST_FILE_SHA256,
            "manifest_sha256": V1_PACKAGE_MANIFEST_SHA256,
        },
        "resume_archive": expected_v1_archive,
        "resume_manifest": expected_v1_manifest,
        "checkpoint_validation": expected_v1_receipt,
        "archive_byte_identity_preserved": True,
        "member_validation_inherited": True,
        "checkpoint_stream_validation_inherited": True,
    }
    receipt = digested(
        {
            "schema": "paper_i_page12_checkpoint_validation_receipt_v2",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "regime_id": regime,
            "resume_round": SOURCE_HORIZON,
            "target_round": TARGET_HORIZON,
            "validation_authority": (
                "sealed_v1_full_stream_validation_plus_v2_byte_identity_v1"
            ),
            "archive": archive_binding,
            "member_count": 3,
            "members": members,
            "checkpoint_sha256": v1_manifest["checkpoint_sha256"],
            "metadata": v1_receipt["metadata"],
            "source_attempt": v1_receipt["source_attempt"],
            "inherited_v1_authority": inherited,
            "worker_validation_scope": (
                "stream_authenticate_all_three_members_then_"
                "strict_resume_replay_v1"
            ),
            "accepted_state_resume_semantic_replay_required": True,
            "ambient_ijson_required": False,
        }
    )
    receipt_path = (
        PACKAGE_DIR / "resume_inputs" / f"{regime}.checkpoint_validation.json"
    )
    _write_json(receipt_path, receipt)
    manifest = digested(
        {
            "schema": "paper_i_page12_pointer_closed_resume_archive_v2",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "regime_id": regime,
            "resume_round": SOURCE_HORIZON,
            "target_round": TARGET_HORIZON,
            "source_kind": "sealed_v1_compact_resume_archive",
            "materialization_kind": "exact_v1_archive_byte_reuse_v1",
            "member_count": 3,
            "members": members,
            "pointer_closed": True,
            "checkpoint_sha256": v1_manifest["checkpoint_sha256"],
            "archive": archive_binding,
            "source_attempt": v1_manifest["source_attempt"],
            "inherited_v1_authority": inherited,
            "checkpoint_validation": file_binding(
                receipt_path, root=PACKAGE_DIR, canonical=True
            ),
        }
    )
    manifest_path = PACKAGE_DIR / "resume_inputs" / f"{regime}.manifest.json"
    _write_json(manifest_path, manifest)
    validate_resume_archive(
        archive_path,
        manifest,
        expected_round=SOURCE_HORIZON,
        checkpoint_validation=receipt,
        verify_archive_members=False,
    )
    return manifest


def _base_protocol_path(regime: str) -> Path:
    return REPO_ROOT / BASE_PROTOCOL_ROOT / f"{source_execution_id(regime)}.json"


def _base_job_path(regime: str) -> Path:
    return BASE_PACKAGE / "jobs" / f"{source_execution_id(regime)}.json"


def _derive_protocols(bundle_manifest_path: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        job_path = _base_job_path(regime)
        _verify_file(job_path, sha256=str(spec["source_job_file_sha256"]))
        source_job = load_json(job_path, label=f"{regime} source job")
        verify_self_digest(source_job, label=f"{regime} source job")
        if source_job["sha256"] != spec["source_job_sha256"]:
            raise PackageContractError(f"Source job digest drifted: {regime}")
        protocol_path = _base_protocol_path(regime)
        source_protocol = load_json(protocol_path, label=f"{regime} protocol")
        verify_self_digest(source_protocol, label=f"{regime} protocol")
        if (
            source_protocol["sha256"] != spec["source_protocol_sha256"]
            or source_protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or source_protocol.get("horizon") != SOURCE_HORIZON
        ):
            raise PackageContractError(f"Source protocol drifted: {regime}")
        target = (
            PACKAGE_DIR / "bundle/protocols" / f"{execution_id(regime)}.json"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(PACKAGE_DIR / "derive_protocol.py"),
                "--base-package",
                str(BASE_PACKAGE),
                "--base-job",
                str(job_path),
                "--bundle-manifest",
                str(bundle_manifest_path),
                "--execution-id",
                execution_id(regime),
                "--regime-id",
                regime,
                "--target-horizon",
                str(TARGET_HORIZON),
                "--output",
                str(target),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise PackageContractError(
                f"Protocol derivation failed for {regime}: {completed.stderr}"
            )
        protocol = load_json(target, label=f"{regime} target protocol")
        verify_self_digest(protocol, label=f"{regime} target protocol")
        if (
            protocol.get("horizon") != TARGET_HORIZON
            or protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or protocol.get("algorithm_id") != ALGORITHM_ID
            or protocol.get("adapter_id") != CANDIDATE_ADAPTER_ID
        ):
            raise PackageContractError(f"Derived protocol drifted: {regime}")
        bindings.append(
            {
                "execution_id": execution_id(regime),
                **file_binding(target, root=PACKAGE_DIR, canonical=True),
            }
        )
    return bindings


def _expected_artifacts(execution: str) -> dict[str, str]:
    root = f"runs/{execution}"
    return {
        role: f"{root}/{suffix}"
        for role, suffix in {
            "execution_manifest": "execution_manifest.json",
            "checkpoint": "checkpoints/current.json",
            "estimator_ledger": "result/estimator_ledger.json",
            "result": "result/result.json",
            "summary": "summary/summary.json",
        }.items()
    }


def build() -> dict[str, Any]:
    if any((PACKAGE_DIR / path).exists() for path in GENERATED_TARGETS):
        raise FileExistsError("Refusing to overwrite the sealed local package.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise PackageContractError(f"Missing control file: {name}")

    v1_manifest_path = V1_PACKAGE / "package_manifest.json"
    _verify_file(
        v1_manifest_path,
        sha256=V1_PACKAGE_MANIFEST_FILE_SHA256,
        size_bytes=9223,
    )
    v1_package_manifest = load_json(
        v1_manifest_path, label="sealed v1 continuation package"
    )
    if (
        verify_self_digest(
            v1_package_manifest, label="sealed v1 continuation package"
        )
        != V1_PACKAGE_MANIFEST_SHA256
        or v1_package_manifest.get("package_id")
        != "paper_i_page12_strong_holstein_r70_accepted_continuations_20260810_v1_local"
        or v1_package_manifest.get("row_count") != 3
        or v1_package_manifest.get("route_contract_sha256")
        != ROUTE_CONTRACT_SHA256
    ):
        raise PackageContractError("Sealed v1 package authority drifted.")

    base_manifest_path = BASE_PACKAGE / "package_manifest.json"
    _verify_file(base_manifest_path, sha256=BASE_PACKAGE_MANIFEST_FILE_SHA256)
    base_manifest = load_json(base_manifest_path, label="base package")
    if verify_self_digest(base_manifest, label="base package") != BASE_PACKAGE_MANIFEST_SHA256:
        raise PackageContractError("Base Page-12 package identity drifted.")
    base_source = BASE_PACKAGE / "source/source_locked.tar.gz"
    _verify_file(base_source, sha256=BASE_SOURCE_ARCHIVE_SHA256)
    base_source_manifest_path = BASE_PACKAGE / "source/source_archive_manifest.json"
    _verify_file(
        base_source_manifest_path, sha256=BASE_SOURCE_MANIFEST_FILE_SHA256
    )
    base_source_manifest = load_json(
        base_source_manifest_path, label="base source manifest"
    )
    if verify_self_digest(base_source_manifest, label="base source manifest") != BASE_SOURCE_MANIFEST_SHA256:
        raise PackageContractError("Base source manifest identity drifted.")
    base_locks_path = (
        BASE_PACKAGE
        / "bundle_materialization/"
        "ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_phase23_"
        "no_lanes_cap24_tau1em4_r50_v1/source_locks.json"
    )
    _verify_file(base_locks_path, sha256=BASE_SOURCE_LOCKS_FILE_SHA256)
    base_locks = load_json(base_locks_path, label="base source locks")
    if verify_self_digest(base_locks, label="base source locks") != BASE_SOURCE_LOCKS_SHA256:
        raise PackageContractError("Base source-lock identity drifted.")

    _copy_exact(base_source, PACKAGE_DIR / "source/source_locked.tar.gz")
    _copy_exact(
        base_source_manifest_path,
        PACKAGE_DIR / "source/source_archive_manifest.json",
    )
    _copy_exact(base_locks_path, PACKAGE_DIR / "bundle/source_locks.json")
    controller_overlay = PACKAGE_DIR / "source_overlay" / CONTROLLER_RELATIVE_PATH
    _verify_file(controller_overlay, sha256=CONTROLLER_AFTER_SHA256)
    resume_overlay = PACKAGE_DIR / "source_overlay" / RESUME_RELATIVE_PATH
    _verify_file(
        resume_overlay,
        sha256=RESUME_AFTER_SHA256,
        size_bytes=RESUME_AFTER_SIZE_BYTES,
    )
    controller_rows = [
        row
        for row in base_source_manifest.get("members", [])
        if isinstance(row, Mapping) and row.get("path") == CONTROLLER_RELATIVE_PATH
    ]
    if len(controller_rows) != 1 or controller_rows[0].get("sha256") != CONTROLLER_BEFORE_SHA256:
        raise PackageContractError("Base controller source binding drifted.")
    resume_rows = [
        row
        for row in base_source_manifest.get("members", [])
        if isinstance(row, Mapping) and row.get("path") == RESUME_RELATIVE_PATH
    ]
    if (
        len(resume_rows) != 1
        or resume_rows[0].get("sha256") != RESUME_BEFORE_SHA256
    ):
        raise PackageContractError("Base resume source binding drifted.")
    controller_regression_path = (
        REPO_ROOT / "test/test_static_adapt_sr_snake_controller.py"
    )
    resume_regression_path = (
        REPO_ROOT / "test/test_static_adapt_resume_insertion_integrity.py"
    )
    composition = digested(
        {
            "schema": "paper_i_page12_r70_local_source_composition_v2",
            "status": "passed",
            "base_archive": file_binding(
                PACKAGE_DIR / "source/source_locked.tar.gz", root=PACKAGE_DIR
            ),
            "base_archive_manifest": file_binding(
                PACKAGE_DIR / "source/source_archive_manifest.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "base_implementation_source_inventory_sha256": base_source_manifest[
                "implementation_source_inventory_sha256"
            ],
            "operational_overlays": [
                {
                    "repair_id": CONTROLLER_REPAIR_ID,
                    "path": CONTROLLER_RELATIVE_PATH,
                    "before_sha256": CONTROLLER_BEFORE_SHA256,
                    "after": file_binding(
                        controller_overlay, root=PACKAGE_DIR
                    ),
                    "semantic_scope": "accepted_energy_roundoff_only",
                    "absolute_tolerance": (
                        "128*ulp(max(1,abs(E1),abs(E2)))"
                    ),
                    "all_non_energy_fields_exact": True,
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                    "regression": {
                        "nodeid": CONTROLLER_REGRESSION,
                        "path": controller_regression_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "sha256": sha256_file(controller_regression_path),
                    },
                },
                {
                    "repair_id": RESUME_REPAIR_ID,
                    "path": RESUME_RELATIVE_PATH,
                    "before_sha256": RESUME_BEFORE_SHA256,
                    "after": file_binding(resume_overlay, root=PACKAGE_DIR),
                    "semantic_scope": (
                        "authenticated_phase0_to_phase1_resume_closure_only"
                    ),
                    "phase0_full_population_authentication_preserved": True,
                    "phase1_full_population_authentication_preserved": True,
                    "phase0_phase1_binding": (
                        "ordered_domain_record_pool_index_insertion_position_"
                        "position_class_v1"
                    ),
                    "legacy_non_phase0_resume_closure_preserved": True,
                    "actual_page12_weak_snapshot_hydration_passed": True,
                    "actual_snapshot_controller_round": SOURCE_HORIZON,
                    "actual_snapshot_route_contract_sha256": (
                        ROUTE_CONTRACT_SHA256
                    ),
                    "scientific_protocol_changed": False,
                    "scientific_settings_changed": [],
                    "regression": {
                        "nodeid": RESUME_REGRESSION,
                        "path": resume_regression_path.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "sha256": sha256_file(resume_regression_path),
                    },
                },
            ],
            "streaming_json_runtime": {
                "distribution": "ijson",
                "version": VENDORED_STREAMING_JSON_VERSION,
                "backend": VENDORED_STREAMING_JSON_BACKEND,
                "ambient_dependency_allowed": False,
                "files": [
                    file_binding(PACKAGE_DIR / path, root=PACKAGE_DIR)
                    for path in VENDORED_STREAMING_JSON_FILES
                ],
            },
            "no_ambient_repo_imports": True,
        }
    )
    composition_path = PACKAGE_DIR / "source/source_composition.json"
    _write_json(composition_path, composition)

    bundle = digested(
        {
            "schema": BUNDLE_MANIFEST_SCHEMA,
            "status": "passed_inert",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "cell_count": 3,
            "cells": [
                {
                    "execution_id": execution_id(str(spec["regime_id"])),
                    "source_execution_id": source_execution_id(
                        str(spec["regime_id"])
                    ),
                    "regime_id": spec["regime_id"],
                    "nph": 7,
                    "source_horizon": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "resume_round": SOURCE_HORIZON,
                }
                for spec in CELL_SPECS
            ],
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "source_locks_sha256": BASE_SOURCE_LOCKS_SHA256,
            "runtime_source_composition_sha256": composition["sha256"],
            "only_scientific_change": {
                "path": "request.execution.stop.maximum_controller_rounds",
                "before": SOURCE_HORIZON,
                "after": TARGET_HORIZON,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    bundle_path = PACKAGE_DIR / "bundle/bundle_manifest.json"
    _write_json(bundle_path, bundle)
    protocol_bindings = _derive_protocols(bundle_path)
    protocol_by_id = {row["execution_id"]: row for row in protocol_bindings}
    resume_manifests = {
        str(spec["regime_id"]): _materialize_resume(
            spec, v1_package_manifest=v1_package_manifest
        )
        for spec in CELL_SPECS
    }

    jobs: list[dict[str, Any]] = []
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        execution = execution_id(regime)
        protocol_binding = protocol_by_id[execution]
        protocol = load_json(
            PACKAGE_DIR / protocol_binding["path"], label=f"{regime} target"
        )
        resume = resume_manifests[regime]
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "bundle_id": BUNDLE_ID,
                "execution_id": execution,
                "source_execution_id": source_execution_id(regime),
                "regime_id": regime,
                "nph": 7,
                "algorithm_id": ALGORITHM_ID,
                "route_id": ROUTE_ID,
                "route_contract_sha256": ROUTE_CONTRACT_SHA256,
                "route_profile": TARGET_ROUTE_PROFILE,
                "candidate_adapter_id": CANDIDATE_ADAPTER_ID,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "source_horizon": SOURCE_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "resume_round": SOURCE_HORIZON,
                "source_protocol_sha256": spec["source_protocol_sha256"],
                "protocol": protocol_binding,
                "protocol_sha256": protocol["sha256"],
                "resume_archive": resume["archive"],
                "resume_manifest": file_binding(
                    PACKAGE_DIR / "resume_inputs" / f"{regime}.manifest.json",
                    root=PACKAGE_DIR,
                    canonical=True,
                ),
                "checkpoint_validation": resume["checkpoint_validation"],
                "checkpoint_sha256": resume["checkpoint_sha256"],
                "runtime_source_composition_sha256": composition["sha256"],
                "resources": dict(RESOURCE_ENVELOPE),
                "expected_artifacts": _expected_artifacts(execution),
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        job_path = PACKAGE_DIR / "jobs" / f"{execution}.json"
        _write_json(job_path, job)
        jobs.append(job)

    plan = digested(
        {
            "schema": "paper_i_page12_strong_r70_local_plan_v2",
            "status": "passed_inert",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 3,
            "execution_ids": list(expected_execution_ids()),
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "resume_rounds": {
                str(spec["regime_id"]): SOURCE_HORIZON for spec in CELL_SPECS
            },
            "max_concurrency": 1,
            "execution_target": EXECUTION_TARGET,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    plan_path = PACKAGE_DIR / "execution_plan.json"
    _write_json(plan_path, plan)
    manifest = digested(
        {
            "schema": PACKAGE_MANIFEST_SCHEMA,
            "status": "passed_inert_three_authenticated_continuations",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "bundle_id": BUNDLE_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "row_count": 3,
            "execution_ids": list(expected_execution_ids()),
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "base_package": {
                "path": BASE_PACKAGE_RELATIVE.as_posix(),
                "manifest_file_sha256": BASE_PACKAGE_MANIFEST_FILE_SHA256,
                "manifest_sha256": BASE_PACKAGE_MANIFEST_SHA256,
            },
            "inherited_v1_continuation_package": {
                "path": V1_PACKAGE_RELATIVE.as_posix(),
                "manifest_file_sha256": V1_PACKAGE_MANIFEST_FILE_SHA256,
                "manifest_sha256": V1_PACKAGE_MANIFEST_SHA256,
                "resume_archive_byte_identity_required": True,
            },
            "runtime_source_composition": file_binding(
                composition_path, root=PACKAGE_DIR, canonical=True
            ),
            "bundle_manifest": file_binding(
                bundle_path, root=PACKAGE_DIR, canonical=True
            ),
            "source_locks": file_binding(
                PACKAGE_DIR / "bundle/source_locks.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "protocols": protocol_bindings,
            "resume_inputs": [
                {
                    "regime_id": regime,
                    "archive": resume_manifests[regime]["archive"],
                    "manifest": file_binding(
                        PACKAGE_DIR / "resume_inputs" / f"{regime}.manifest.json",
                        root=PACKAGE_DIR,
                        canonical=True,
                    ),
                    "checkpoint_validation": resume_manifests[regime][
                        "checkpoint_validation"
                    ],
                }
                for regime in (str(spec["regime_id"]) for spec in CELL_SPECS)
            ],
            "jobs": [
                {
                    "execution_id": job["execution_id"],
                    **file_binding(
                        PACKAGE_DIR / "jobs" / f"{job['execution_id']}.json",
                        root=PACKAGE_DIR,
                        canonical=True,
                    ),
                }
                for job in jobs
            ],
            "execution_plan": file_binding(
                plan_path, root=PACKAGE_DIR, canonical=True
            ),
            "control_files": [
                file_binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "max_concurrency": 1,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    manifest_path = PACKAGE_DIR / "package_manifest.json"
    _write_json(manifest_path, manifest)

    request = digested(
        {
            "schema": "paper_i_page12_strong_r70_local_activation_request_v2",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest_sha256": manifest["sha256"],
            "requested_execution_ids": list(expected_execution_ids()),
            "authority_scope": "three_page12_strong_continuations_to_round_70",
            "authorization_kind": "explicit_user_local_execution_authority",
            "execution_target": EXECUTION_TARGET,
            "execution_authorized": True,
            "submission_authorized": False,
            "paper_evidence_adoption_authorized": False,
            "submitted": False,
        }
    )
    request_path = PACKAGE_DIR / "activation/activation_request.json"
    _write_json(request_path, request)
    auth_bindings: list[dict[str, Any]] = []
    for job in jobs:
        authority = digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "package_manifest_sha256": manifest["sha256"],
                "activation_request_sha256": request["sha256"],
                "job_spec_sha256": job["sha256"],
                "protocol_sha256": job["protocol_sha256"],
                "resume_archive_sha256": job["resume_archive"]["sha256"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                "checkpoint_validation_sha256": job[
                    "checkpoint_validation"
                ]["canonical_sha256"],
                "runtime_source_composition_sha256": composition["sha256"],
                "authorization_kind": "explicit_user_local_execution_authority",
                "execution_target": EXECUTION_TARGET,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_evidence_adoption_authorized": False,
                "submitted": False,
            }
        )
        auth_path = (
            PACKAGE_DIR
            / "activation/authorizations"
            / f"{job['execution_id']}.json"
        )
        _write_json(auth_path, authority)
        auth_bindings.append(
            {
                "execution_id": job["execution_id"],
                **file_binding(auth_path, root=PACKAGE_DIR, canonical=True),
            }
        )
    activation = digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "status": "passed_local_activation_prepared_no_execution",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest_sha256": manifest["sha256"],
            "activation_request": file_binding(
                request_path, root=PACKAGE_DIR, canonical=True
            ),
            "authorizations": auth_bindings,
            "authorization_count": 3,
            "execution_target": EXECUTION_TARGET,
            "max_concurrency": 1,
            "page13_completion_gate_required": True,
            "pre_run_capacity_gate_required": True,
            "execution_authorized": True,
            "submission_authorized": False,
            "launch_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / "activation/activation_manifest.json", activation)
    return {
        "status": "passed_local_activation_prepared_no_execution",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "activation_manifest_sha256": activation["sha256"],
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "row_count": 3,
        "submission_authorized": False,
        "scientific_execution_performed": False,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
