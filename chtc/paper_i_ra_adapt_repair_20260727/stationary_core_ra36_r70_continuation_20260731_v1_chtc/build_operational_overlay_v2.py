#!/usr/bin/env python3
"""Build the explicit retention-v2 operational overlay.

The verified v1 compact resume archives are reused byte-for-byte.  This
builder writes only new v2 metadata/jobs plus a deterministic one-member core
source repair.  It never submits, stages, authorizes, or contacts CHTC.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from operational_overlay_v2_contract import (  # noqa: E402
    AUTHORIZATION_V2_SCHEMA,
    BUILD_RECEIPT_V2_NAME,
    BUILD_RECEIPT_V2_SCHEMA,
    CHECKPOINT_MEMBER,
    COLLISION_EVIDENCE_NAME,
    COLLISION_EVIDENCE_SCHEMA,
    EFFECTIVE_ALWAYS_FAMILY,
    EFFECTIVE_CORE_FAMILY,
    EFFECTIVE_EXECUTION_CONTRACT_SCHEMA,
    EFFECTIVE_FAMILY_BY_BASE_FAMILY,
    EFFECTIVE_SOURCES_DIR,
    EFFECTIVE_SOURCES_NAME,
    EFFECTIVE_SOURCES_SCHEMA,
    EXECUTION_PLAN_V2_NAME,
    EXECUTION_PLAN_V2_SCHEMA,
    JOBS_V2_DIR,
    JOB_V2_SCHEMA,
    OVERLAY_CONTROL_FILES,
    OVERLAY_GENERATED_FILES,
    OVERLAY_ID,
    OVERLAY_MANIFEST_NAME,
    OVERLAY_MANIFEST_SCHEMA,
    OVERLAY_PACKAGE_ID,
    PARENT_CHECKPOINT_SHA256,
    QUEUE_V2_NAME,
    REPAIRED_CHECKPOINT_SHA256,
    SOURCE_LOCK_AUDIT_V2_NAME,
    SOURCE_LOCK_AUDIT_V2_SCHEMA,
    build_effective_execution_contract,
    effective_contract_sha256,
)
from package_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CELL_COUNT,
    COLLISION_QUEUE_RELATIVE,
    COLLISION_STATE_SNAPSHOT_RELATIVE,
    COLLISION_STATUS_NAME,
    COLLISION_SUBMISSION_RECEIPT_RELATIVE,
    EXECUTION_PLAN_NAME,
    FRESH_COUNT,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    RESUME_COUNT,
    RESUME_INPUTS_NAME,
    RUN_CLASS,
    SOURCE_ARCHIVES_NAME,
    SOURCE_HORIZON,
    SOURCE_LOCK_AUDIT_NAME,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    load_json,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)
from validate_package import validate_package  # noqa: E402


ALWAYS_V2_PACKAGE_RELATIVE = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v2_chtc"
)


def _exclusive_write(
    path: Path, payload: bytes, *, executable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _exclusive_write(path, canonical_json_bytes(payload) + b"\n")


def _exclusive_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise PackageContractError(
            f"Refusing to overwrite: {destination}"
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with source.open("rb") as input_stream:
            with temporary.open("xb") as output_stream:
                shutil.copyfileobj(
                    input_stream, output_stream, length=1024 * 1024
                )
                output_stream.flush()
                os.fsync(output_stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _file_binding(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() == PACKAGE_DIR.resolve()
    ):
        raise PackageContractError(f"Unsafe binding path: {path}")
    return {
        "path": path.relative_to(PACKAGE_DIR).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _json_binding(path: Path) -> dict[str, Any]:
    payload = load_json(path, label=f"{path.name} binding")
    verify_self_digest(payload, label=f"{path.name} binding")
    return {
        **_file_binding(path),
        "canonical_sha256": payload["sha256"],
    }


def _source_members(
    *,
    archive_path: Path,
    manifest_path: Path,
    label: str,
) -> tuple[
    list[str],
    dict[str, bytes],
    dict[str, dict[str, Any]],
]:
    manifest = load_json(manifest_path, label=f"{label} manifest")
    verify_self_digest(manifest, label=f"{label} manifest")
    archive_binding = manifest.get("archive")
    rows = manifest.get("members")
    if (
        not isinstance(archive_binding, Mapping)
        or not isinstance(rows, list)
        or archive_binding.get("sha256")
        != sha256_file(archive_path)
        or int(archive_binding.get("size_bytes", -1))
        != archive_path.stat().st_size
    ):
        raise PackageContractError(
            f"{label} archive binding drifted."
        )
    declared = {
        str(row["path"]): dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
    }
    if len(declared) != len(rows):
        raise PackageContractError(
            f"{label} source-member index duplicates."
        )
    order: list[str] = []
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label=f"{label} member path"
            ).as_posix()
            if (
                relative not in declared
                or relative in payloads
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe {label} member: {relative}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable {label} member: {relative}"
                )
            payload = stream.read()
            row = declared[relative]
            if (
                hashlib.sha256(payload).hexdigest()
                != row.get("sha256")
                or len(payload)
                != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"{label} member drifted: {relative}"
                )
            row["mode"] = int(member.mode)
            order.append(relative)
            payloads[relative] = payload
    if set(order) != set(declared):
        raise PackageContractError(
            f"{label} member closure drifted."
        )
    return order, payloads, declared


def _write_source_archive(
    *,
    path: Path,
    order: list[str],
    payloads: Mapping[str, bytes],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    if path.exists() or path.is_symlink():
        raise PackageContractError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for relative in order:
                        payload = payloads[relative]
                        info = tarfile.TarInfo(relative)
                        info.size = len(payload)
                        info.mode = int(metadata[relative]["mode"])
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _assert_one_member_delta(
    *,
    parent_archive: Path,
    parent_manifest: Path,
    effective_archive: Path,
    effective_manifest: Path,
    label: str,
) -> dict[str, Any]:
    parent_order, parent_payloads, parent_rows = _source_members(
        archive_path=parent_archive,
        manifest_path=parent_manifest,
        label=f"{label} parent",
    )
    effective_order, effective_payloads, effective_rows = (
        _source_members(
            archive_path=effective_archive,
            manifest_path=effective_manifest,
            label=f"{label} effective",
        )
    )
    changed = [
        name
        for name in parent_order
        if parent_payloads[name] != effective_payloads.get(name)
    ]
    protocol_members = [
        name
        for name in parent_order
        if "/protocols/" in name and name.endswith(".json")
    ]
    if (
        effective_order != parent_order
        or changed != [CHECKPOINT_MEMBER]
        or set(effective_payloads) != set(parent_payloads)
        or {
            name: int(row["mode"])
            for name, row in effective_rows.items()
        }
        != {
            name: int(row["mode"])
            for name, row in parent_rows.items()
        }
        or hashlib.sha256(
            parent_payloads[CHECKPOINT_MEMBER]
        ).hexdigest()
        != PARENT_CHECKPOINT_SHA256
        or hashlib.sha256(
            effective_payloads[CHECKPOINT_MEMBER]
        ).hexdigest()
        != REPAIRED_CHECKPOINT_SHA256
        or not protocol_members
        or any(
            parent_payloads[name] != effective_payloads[name]
            for name in protocol_members
        )
    ):
        raise PackageContractError(
            f"{label} is not the exact approved one-member delta."
        )
    return {
        "changed_members": changed,
        "parent_member_count": len(parent_order),
        "effective_member_count": len(effective_order),
        "protocol_member_count": len(protocol_members),
        "protocol_members_byte_identical": True,
        "ordered_member_paths_identical": True,
        "member_modes_identical": True,
        "scientific_settings_changed": [],
    }


def _derive_core_source(
    *,
    parent_archive: Path,
    parent_manifest: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    order, parent_payloads, rows = _source_members(
        archive_path=parent_archive,
        manifest_path=parent_manifest,
        label="core-v8 parent",
    )
    repaired = (
        repo_root / CHECKPOINT_MEMBER
    ).read_bytes()
    if (
        hashlib.sha256(
            parent_payloads[CHECKPOINT_MEMBER]
        ).hexdigest()
        != PARENT_CHECKPOINT_SHA256
        or hashlib.sha256(repaired).hexdigest()
        != REPAIRED_CHECKPOINT_SHA256
    ):
        raise PackageContractError(
            "Approved core checkpoint-retention source drifted."
        )
    payloads = dict(parent_payloads)
    payloads[CHECKPOINT_MEMBER] = repaired
    root = (
        PACKAGE_DIR
        / EFFECTIVE_SOURCES_DIR
        / EFFECTIVE_CORE_FAMILY
    )
    archive_path = root / "source_locked.tar.gz"
    _write_source_archive(
        path=archive_path,
        order=order,
        payloads=payloads,
        metadata=rows,
    )
    member_rows = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                payloads[relative]
            ).hexdigest(),
            "size_bytes": len(payloads[relative]),
            "mode": int(rows[relative]["mode"]),
        }
        for relative in order
    ]
    manifest = digested(
        {
            "schema": (
                "paper_i_checkpoint_sidecar_retention_source_archive_v2"
            ),
            "operational_package_id": OVERLAY_PACKAGE_ID,
            "scientific_parent_package_id": PACKAGE_ID,
            "status": "passed",
            "archive": {
                "path": "source_locked.tar.gz",
                "sha256": sha256_file(archive_path),
                "size_bytes": archive_path.stat().st_size,
            },
            "parent_archive": {
                **_file_binding(parent_archive),
            },
            "member_count": len(member_rows),
            "members": member_rows,
            "deterministic_archive": {
                "gzip_mtime": 0,
                "tar_member_mtime": 0,
                "uid": 0,
                "gid": 0,
                "ordered_as_parent": True,
                "parent_member_modes_preserved": True,
            },
        }
    )
    manifest_path = root / "source_archive_manifest.json"
    _write_json(manifest_path, manifest)
    delta = digested(
        {
            "schema": (
                "paper_i_checkpoint_sidecar_retention_source_delta_v2"
            ),
            "operational_package_id": OVERLAY_PACKAGE_ID,
            "scientific_parent_package_id": PACKAGE_ID,
            "parent_source_archive_sha256": sha256_file(
                parent_archive
            ),
            "repaired_source_archive_sha256": sha256_file(
                archive_path
            ),
            "parent_member_count": len(order),
            "repaired_member_count": len(order),
            "unchanged_member_count": len(order) - 1,
            "changed_member_count": 1,
            "changed_members": [
                {
                    "path": CHECKPOINT_MEMBER,
                    "parent_sha256": PARENT_CHECKPOINT_SHA256,
                    "parent_size_bytes": len(
                        parent_payloads[CHECKPOINT_MEMBER]
                    ),
                    "parent_mode": int(
                        rows[CHECKPOINT_MEMBER]["mode"]
                    ),
                    "repaired_sha256": (
                        REPAIRED_CHECKPOINT_SHA256
                    ),
                    "repaired_size_bytes": len(repaired),
                    "repaired_mode": int(
                        rows[CHECKPOINT_MEMBER]["mode"]
                    ),
                    "classification": (
                        "observation_only_authenticated_predecessor_"
                        "sidecar_retirement"
                    ),
                    "scientific_protocol_change": False,
                    "controller_semantics_change": False,
                }
            ],
            "protocol_members_byte_identical": True,
            "non_checkpoint_members_byte_identical": True,
            "ordered_member_paths_identical": True,
            "member_modes_identical": True,
            "scientific_settings_changed": [],
            "status": "passed",
        }
    )
    delta_path = root / "source_delta_receipt.json"
    _write_json(delta_path, delta)
    proof = _assert_one_member_delta(
        parent_archive=parent_archive,
        parent_manifest=parent_manifest,
        effective_archive=archive_path,
        effective_manifest=manifest_path,
        label="core-v8 retention-v2",
    )
    family = {
        "effective_family": EFFECTIVE_CORE_FAMILY,
        "supersedes_base_families": ["stationary_core_v11"],
        "parent_archive": _file_binding(parent_archive),
        "parent_manifest": _json_binding(parent_manifest),
        "effective_archive": _file_binding(archive_path),
        "effective_manifest": _json_binding(manifest_path),
        "delta_receipt": _json_binding(delta_path),
        "checkpoint_member": CHECKPOINT_MEMBER,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "effective_checkpoint_sha256": (
            REPAIRED_CHECKPOINT_SHA256
        ),
        "delta_proof": proof,
        "runtime_source_kind": (
            "sealed_one_member_operational_derivation"
        ),
    }
    return family, delta


def _bind_always_source(
    *,
    source_archives: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    v1 = source_archives["families"]["always_factorial_v1"]
    v2 = source_archives["families"]["always_factorial_v2"]
    parent_archive = PACKAGE_DIR / v1["packaged_archive"]["path"]
    parent_manifest = PACKAGE_DIR / v1["packaged_manifest"]["path"]
    effective_archive = PACKAGE_DIR / v2["packaged_archive"]["path"]
    effective_manifest = PACKAGE_DIR / v2["packaged_manifest"]["path"]
    source_delta = (
        repo_root
        / ALWAYS_V2_PACKAGE_RELATIVE
        / "source_delta_receipt.json"
    )
    delta_root = (
        PACKAGE_DIR
        / EFFECTIVE_SOURCES_DIR
        / EFFECTIVE_ALWAYS_FAMILY
    )
    delta_path = delta_root / "source_delta_receipt.json"
    _exclusive_copy(source_delta, delta_path)
    delta = load_json(delta_path, label="always-v2 source delta")
    verify_self_digest(delta, label="always-v2 source delta")
    proof = _assert_one_member_delta(
        parent_archive=parent_archive,
        parent_manifest=parent_manifest,
        effective_archive=effective_archive,
        effective_manifest=effective_manifest,
        label="always-factorial retention-v2",
    )
    changed = delta.get("changed_members")
    if (
        delta.get("schema")
        != "paper_i_checkpoint_sidecar_retention_source_delta_v2"
        or delta.get("parent_source_archive_sha256")
        != sha256_file(parent_archive)
        or delta.get("repaired_source_archive_sha256")
        != sha256_file(effective_archive)
        or delta.get("scientific_settings_changed") != []
        or not isinstance(changed, list)
        or len(changed) != 1
        or changed[0].get("path") != CHECKPOINT_MEMBER
        or changed[0].get("parent_sha256")
        != PARENT_CHECKPOINT_SHA256
        or changed[0].get("repaired_sha256")
        != REPAIRED_CHECKPOINT_SHA256
    ):
        raise PackageContractError(
            "Existing always-v2 source delta drifted."
        )
    family = {
        "effective_family": EFFECTIVE_ALWAYS_FAMILY,
        "supersedes_base_families": [
            "always_factorial_v1",
            "always_factorial_v2",
        ],
        "parent_archive": _file_binding(parent_archive),
        "parent_manifest": _json_binding(parent_manifest),
        "effective_archive": _file_binding(effective_archive),
        "effective_manifest": _json_binding(effective_manifest),
        "delta_receipt": _json_binding(delta_path),
        "checkpoint_member": CHECKPOINT_MEMBER,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "effective_checkpoint_sha256": (
            REPAIRED_CHECKPOINT_SHA256
        ),
        "delta_proof": proof,
        "runtime_source_kind": (
            "existing_sealed_always_factorial_v2"
        ),
    }
    return family, delta


def _effective_sources(
    *,
    source_archives: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    core = source_archives["families"]["stationary_core_v11"]
    core_family, _core_delta = _derive_core_source(
        parent_archive=(
            PACKAGE_DIR / core["packaged_archive"]["path"]
        ),
        parent_manifest=(
            PACKAGE_DIR / core["packaged_manifest"]["path"]
        ),
        repo_root=repo_root,
    )
    always_family, _always_delta = _bind_always_source(
        source_archives=source_archives, repo_root=repo_root
    )
    return digested(
        {
            "schema": EFFECTIVE_SOURCES_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "base_package_id": PACKAGE_ID,
            "status": "passed",
            "family_count": 2,
            "families": {
                EFFECTIVE_CORE_FAMILY: core_family,
                EFFECTIVE_ALWAYS_FAMILY: always_family,
            },
            "base_to_effective_family": dict(
                EFFECTIVE_FAMILY_BY_BASE_FAMILY
            ),
            "changed_source_members": [CHECKPOINT_MEMBER],
            "protocol_members_byte_identical": True,
            "scientific_settings_changed": [],
            "controller_semantics_changed": False,
            "observation_retention_only": True,
        }
    )


def _collision_evidence(repo_root: Path) -> dict[str, Any]:
    collision = load_json(
        PACKAGE_DIR / COLLISION_STATUS_NAME,
        label="baseline collision status",
    )
    verify_self_digest(collision, label="baseline collision status")
    receipt_path = repo_root / COLLISION_SUBMISSION_RECEIPT_RELATIVE
    receipt = load_json(receipt_path, label="collision receipt")
    verify_self_digest(receipt, label="collision receipt")
    queue_path = repo_root / COLLISION_QUEUE_RELATIVE
    state_path = repo_root / COLLISION_STATE_SNAPSHOT_RELATIVE
    queue_binding = (
        receipt.get("bindings", {}).get("queue_manifest")
    )
    baseline_queue = (
        collision.get("bindings", {}).get("queue")
    )
    if (
        receipt.get("cluster_id") != 9397758
        or not isinstance(queue_binding, Mapping)
        or not isinstance(baseline_queue, Mapping)
        or queue_binding.get("path")
        != COLLISION_QUEUE_RELATIVE
        or queue_binding.get("sha256")
        != baseline_queue.get("sha256")
        != sha256_file(queue_path)
        or int(queue_binding.get("size_bytes", -1))
        != queue_path.stat().st_size
        or collision.get("blocking") is not True
        or collision.get("external_state_revalidation_required")
        is not True
    ):
        raise PackageContractError(
            "Collision receipt/queue binding drifted."
        )
    return digested(
        {
            "schema": COLLISION_EVIDENCE_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "status": "blocked_stale_local_evidence",
            "blocking": True,
            "cluster_id": 9397758,
            "proc_ids": list(range(9)),
            "submission_receipt": {
                "path": COLLISION_SUBMISSION_RECEIPT_RELATIVE,
                "sha256": sha256_file(receipt_path),
                "size_bytes": receipt_path.stat().st_size,
                "canonical_sha256": receipt["sha256"],
            },
            "bound_queue": {
                "path": COLLISION_QUEUE_RELATIVE,
                "sha256": sha256_file(queue_path),
                "size_bytes": queue_path.stat().st_size,
            },
            "local_state_snapshot": {
                "path": COLLISION_STATE_SNAPSHOT_RELATIVE,
                "sha256": sha256_file(state_path),
                "size_bytes": state_path.stat().st_size,
            },
            "baseline_collision_status": _json_binding(
                PACKAGE_DIR / COLLISION_STATUS_NAME
            ),
            "external_state_revalidation_required": True,
            "fresh_execution_requires_sealed_clearance": True,
            "submission_ready": False,
            "may_submit": False,
            "may_supersede_predecessors": False,
            "may_remove_predecessors": False,
        }
    )


def _draft_jobs(
    *,
    baseline_manifest: Mapping[str, Any],
    effective_sources: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for binding in baseline_manifest["jobs"]:
        path = PACKAGE_DIR / binding["path"]
        base = load_json(path, label=f"{binding['execution_id']} job")
        verify_self_digest(base, label=f"{binding['execution_id']} job")
        base_family = str(base["source_family"])
        effective_family = EFFECTIVE_FAMILY_BY_BASE_FAMILY[
            base_family
        ]
        family = effective_sources["families"][effective_family]
        draft = dict(base)
        draft["parent_job"] = {
            **_file_binding(path),
            "canonical_sha256": base["sha256"],
        }
        draft["base_package_id"] = PACKAGE_ID
        draft["package_id"] = OVERLAY_PACKAGE_ID
        draft["overlay_id"] = OVERLAY_ID
        draft["effective_source_family"] = effective_family
        draft["scientific_parent_source_archive"] = dict(
            base["source_archive"]
        )
        draft["scientific_parent_source_archive_manifest"] = (
            dict(base["source_archive_manifest"])
        )
        draft["effective_source_archive"] = dict(
            family["effective_archive"]
        )
        draft["effective_source_archive_manifest"] = dict(
            family["effective_manifest"]
        )
        draft["effective_source_delta_receipt"] = dict(
            family["delta_receipt"]
        )
        # The sealed v1 helper consumes these conventional names.
        draft["source_archive"] = dict(
            family["effective_archive"]
        )
        draft["source_archive_manifest"] = dict(
            family["effective_manifest"]
        )
        result[str(base["execution_id"])] = draft
    if len(result) != CELL_COUNT:
        raise PackageContractError("Baseline job matrix drifted.")
    return result


def _derive_protocols_for_family(
    *,
    family: str,
    jobs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    script = r"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

package = Path(sys.argv[1]).resolve()
jobs_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
spec = importlib.util.spec_from_file_location(
    "r70_overlay_source_runtime", package / "run_cell.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
with tempfile.TemporaryDirectory(
    prefix="r70-overlay-derived-protocols."
) as raw:
    source = Path(raw) / "source"
    module._extract_source(jobs[0], source)
    module._activate_source_root(source)
    result = {}
    for job in jobs:
        protocol, _problem, delta = module._derived_protocol(
            job=job, source_root=source
        )
        result[job["execution_id"]] = {
            "protocol": protocol.to_dict(),
            "delta": delta,
        }
    output_path.write_text(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
"""
    with tempfile.TemporaryDirectory(
        prefix=f"r70-{family}-contracts."
    ) as raw:
        root = Path(raw)
        jobs_path = root / "jobs.json"
        output_path = root / "derived.json"
        jobs_path.write_bytes(canonical_json_bytes(jobs) + b"\n")
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                script,
                str(PACKAGE_DIR),
                str(jobs_path),
                str(output_path),
            ],
            cwd=repo_root_from_script(__file__),
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise PackageContractError(
                f"Derived-protocol preflight failed for {family}: "
                f"{completed.stderr}"
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or set(result) != {
        str(job["execution_id"]) for job in jobs
    }:
        raise PackageContractError(
            f"Derived-protocol closure failed for {family}."
        )
    return result


def _derive_all_protocols(
    drafts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {
        EFFECTIVE_CORE_FAMILY: [],
        EFFECTIVE_ALWAYS_FAMILY: [],
    }
    for job in drafts.values():
        grouped[str(job["effective_source_family"])].append(job)
    result: dict[str, Any] = {}
    for family, jobs in grouped.items():
        observed = _derive_protocols_for_family(
            family=family, jobs=jobs
        )
        overlap = set(result).intersection(observed)
        if overlap:
            raise PackageContractError(
                f"Duplicate derived protocols: {sorted(overlap)}"
            )
        result.update(observed)
        print(
            f"derived {len(observed)} r70 protocols from {family}",
            flush=True,
        )
    if len(result) != CELL_COUNT:
        raise PackageContractError(
            "All-36 derived protocol closure failed."
        )
    return result


def _source_lock_audit_v2(
    *,
    drafts: Mapping[str, Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
    effective_sources: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for execution_id, job in drafts.items():
        contract = contracts[execution_id]
        resume = job.get("resume_input")
        anchor = {
            "kind": (
                "authenticated_round50_checkpoint"
                if resume is not None
                else "unavailable_live_round50_predecessor"
            ),
            "checkpoint_metadata_authenticated": (
                resume is not None
            ),
            "checkpoint_sha256": (
                resume["checkpoint_sha256"]
                if resume is not None
                else None
            ),
            "operator_sequence_digest_available": False,
            "operator_sequence_match_claimed": False,
            "non_swept_settings_diff": [],
        }
        rows.append(
            {
                "execution_id": execution_id,
                "base_execution_id": job["base_execution_id"],
                "effective_source_family": job[
                    "effective_source_family"
                ],
                "source_protocol_sha256": job[
                    "source_protocol"
                ]["canonical_sha256"],
                "derived_protocol_sha256": contract[
                    "scientific_settings"
                ]["derived_protocol_sha256"],
                "effective_execution_contract_sha256": contract[
                    "sha256"
                ],
                "scientific_settings_sha256": contract[
                    "scientific_settings_sha256"
                ],
                "operational_settings_sha256": contract[
                    "operational_settings_sha256"
                ],
                "changed_fields_vs_source": [
                    "maximum_controller_rounds"
                ],
                "changed_serialized_paths_vs_source": [
                    "horizon",
                    (
                        "request.execution.stop."
                        "maximum_controller_rounds"
                    ),
                    (
                        "stopping_rule."
                        "maximum_controller_rounds"
                    ),
                ],
                "source_operational_delta": {
                    "changed_members": [CHECKPOINT_MEMBER],
                    "scientific_settings_changed": [],
                    "effective_source_archive_sha256": job[
                        "effective_source_archive"
                    ]["sha256"],
                    "source_delta_receipt_sha256": job[
                        "effective_source_delta_receipt"
                    ]["canonical_sha256"],
                },
                "non_swept_settings_diff": [],
                "fields_added_by_current_defaults": [],
                "unresolved_source_fields": [],
                "anchor": anchor,
                "status": (
                    "passed_authenticated_checkpoint_metadata_anchor"
                    if resume is not None
                    else "blocked_live_r50_predecessor"
                ),
            }
        )
    return digested(
        {
            "schema": SOURCE_LOCK_AUDIT_V2_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "base_package_id": PACKAGE_ID,
            "run_class": RUN_CLASS,
            "sweep": {
                "variable": "maximum_controller_rounds",
                "source_value": SOURCE_HORIZON,
                "grid": [TARGET_HORIZON],
                "settings_changed": [
                    "maximum_controller_rounds"
                ],
                "scientific_settings_changed_by_operational_overlay": [],
                "fields_added_by_current_defaults": [],
                "unresolved_source_fields": [],
            },
            "effective_sources_sha256": effective_sources["sha256"],
            "planned_rows": rows,
            "anchor": {
                "authenticated_checkpoint_metadata_count": (
                    RESUME_COUNT
                ),
                "blocked_fresh_anchor_count": FRESH_COUNT,
                "operator_sequence_match_claim_count": 0,
                "all_resume_checkpoint_metadata_close": True,
                "all_fresh_rows_blocked": True,
            },
            "status": (
                "blocked_live_r50_predecessors_with_"
                "27_authenticated_checkpoint_metadata_anchors"
            ),
        }
    )


def main() -> int:
    repo_root = repo_root_from_script(__file__)
    for name in OVERLAY_CONTROL_FILES:
        path = PACKAGE_DIR / name
        if not path.is_file() or path.is_symlink():
            raise PackageContractError(
                f"Overlay control file is missing: {name}"
            )
    for name in OVERLAY_GENERATED_FILES:
        path = PACKAGE_DIR / name
        if path.exists() or path.is_symlink():
            raise PackageContractError(
                f"Refusing to overwrite v2 overlay file: {path}"
            )
    for name in (JOBS_V2_DIR, EFFECTIVE_SOURCES_DIR):
        path = PACKAGE_DIR / name
        if path.exists() or path.is_symlink():
            raise PackageContractError(
                f"Refusing to overwrite v2 overlay directory: {path}"
            )
    if (
        (PACKAGE_DIR / "submit.sub").exists()
        or (PACKAGE_DIR / "authority").exists()
    ):
        raise PackageContractError(
            "The inert base package gained submission state."
        )

    baseline_validation = validate_package(
        full_archive_scan=True
    )
    baseline_manifest = load_json(
        PACKAGE_DIR / PACKAGE_MANIFEST_NAME,
        label="baseline package manifest",
    )
    verify_self_digest(
        baseline_manifest, label="baseline package manifest"
    )
    source_archives = load_json(
        PACKAGE_DIR / SOURCE_ARCHIVES_NAME,
        label="baseline source archives",
    )
    verify_self_digest(source_archives, label="baseline source archives")
    resume_inputs = load_json(
        PACKAGE_DIR / RESUME_INPUTS_NAME,
        label="baseline resume inputs",
    )
    verify_self_digest(resume_inputs, label="baseline resume inputs")
    if (
        baseline_validation.get("full_archive_scan") is not True
        or baseline_validation.get("authenticated_resume_count")
        != RESUME_COUNT
        or resume_inputs.get("resume_cell_count") != RESUME_COUNT
    ):
        raise PackageContractError(
            "Baseline compact inputs were not fully verified."
        )

    effective_sources = _effective_sources(
        source_archives=source_archives,
        repo_root=repo_root,
    )
    _write_json(
        PACKAGE_DIR / EFFECTIVE_SOURCES_NAME,
        effective_sources,
    )
    collision_evidence = _collision_evidence(repo_root)
    _write_json(
        PACKAGE_DIR / COLLISION_EVIDENCE_NAME,
        collision_evidence,
    )
    drafts = _draft_jobs(
        baseline_manifest=baseline_manifest,
        effective_sources=effective_sources,
    )
    derived = _derive_all_protocols(drafts)
    contracts = {
        execution_id: build_effective_execution_contract(
            job=job,
            derived_protocol_payload=derived[execution_id][
                "protocol"
            ],
        )
        for execution_id, job in drafts.items()
    }
    for contract in contracts.values():
        effective_contract_sha256(contract)
        if (
            contract.get("schema")
            != EFFECTIVE_EXECUTION_CONTRACT_SCHEMA
        ):
            raise PackageContractError(
                "Effective execution contract schema drifted."
            )

    audit = _source_lock_audit_v2(
        drafts=drafts,
        contracts=contracts,
        effective_sources=effective_sources,
    )
    _write_json(
        PACKAGE_DIR / SOURCE_LOCK_AUDIT_V2_NAME,
        audit,
    )
    plan = digested(
        {
            "schema": EXECUTION_PLAN_V2_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "base_package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "cell_count": CELL_COUNT,
            "authenticated_resume_count": RESUME_COUNT,
            "fresh_count": FRESH_COUNT,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "execution_ids": list(drafts),
            "only_scientific_change": (
                "maximum_controller_rounds_50_to_70"
            ),
            "operational_overlay": {
                "changed_source_members": [CHECKPOINT_MEMBER],
                "scientific_settings_changed": [],
                "controller_semantics_changed": False,
                "observation_retention_only": True,
            },
            "effective_sources_sha256": effective_sources["sha256"],
            "source_lock_audit_sha256": audit["sha256"],
            "collision_evidence_sha256": (
                collision_evidence["sha256"]
            ),
            "effective_execution_contracts": {
                execution_id: {
                    "sha256": contract["sha256"],
                    "scientific_settings_sha256": contract[
                        "scientific_settings_sha256"
                    ],
                    "operational_settings_sha256": contract[
                        "operational_settings_sha256"
                    ],
                }
                for execution_id, contract in contracts.items()
            },
            "submission_blockers": [
                "live_r50_predecessors_9397758_0_through_8",
                "fresh_row_collision_clearances_absent",
                "r70_resource_envelopes_not_demonstrated",
                "execution_authorization_absent",
                "submission_descriptor_intentionally_absent",
            ],
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / EXECUTION_PLAN_V2_NAME, plan)

    jobs_dir = PACKAGE_DIR / JOBS_V2_DIR
    jobs_dir.mkdir()
    job_bindings: list[dict[str, Any]] = []
    for execution_id, draft in drafts.items():
        parent_plan_sha = draft.get("execution_plan_sha256")
        parent_audit_sha = draft.get("source_lock_audit_sha256")
        job_payload = dict(draft)
        job_payload.pop("sha256", None)
        job_payload["schema"] = JOB_V2_SCHEMA
        job_payload["package_id"] = OVERLAY_PACKAGE_ID
        job_payload["base_execution_plan_sha256"] = (
            parent_plan_sha
        )
        job_payload["base_source_lock_audit_sha256"] = (
            parent_audit_sha
        )
        job_payload["base_package_manifest_sha256"] = (
            baseline_manifest["sha256"]
        )
        job_payload["effective_sources_sha256"] = (
            effective_sources["sha256"]
        )
        job_payload["execution_plan_sha256"] = plan["sha256"]
        job_payload["source_lock_audit_sha256"] = audit["sha256"]
        job_payload["collision_evidence_sha256"] = (
            collision_evidence["sha256"]
        )
        job_payload["effective_execution_contract"] = contracts[
            execution_id
        ]
        job_payload["effective_execution_contract_sha256"] = (
            contracts[execution_id]["sha256"]
        )
        job_payload["scientific_settings_sha256"] = contracts[
            execution_id
        ]["scientific_settings_sha256"]
        job_payload["operational_settings_sha256"] = contracts[
            execution_id
        ]["operational_settings_sha256"]
        job_payload["authorization_schema"] = (
            AUTHORIZATION_V2_SCHEMA
        )
        job_payload["collision_clearance_required"] = (
            draft["execution_mode"] == "fresh_0_to_70"
        )
        job_payload["execution_authorized"] = False
        job_payload["submission_authorized"] = False
        job_payload["submission_ready"] = False
        job_payload["submitted"] = False
        job = digested(job_payload)
        path = jobs_dir / f"{execution_id}.json"
        _write_json(path, job)
        job_bindings.append(
            {
                "execution_id": execution_id,
                "path": path.relative_to(PACKAGE_DIR).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "canonical_sha256": job["sha256"],
                "parent_job": dict(draft["parent_job"]),
            }
        )

    queue_lines = [
        "\t".join(
            (
                execution_id,
                str(draft["execution_mode"]),
                str(draft["collision_status"]),
                str(draft["effective_source_family"]),
                str(draft["resources"]["request_cpus"]),
                str(draft["resources"]["request_memory_mb"]),
                str(draft["resources"]["request_disk_mb"]),
                str(draft["resources"]["max_runtime_seconds"]),
            )
        )
        for execution_id, draft in drafts.items()
    ]
    _exclusive_write(
        PACKAGE_DIR / QUEUE_V2_NAME,
        ("\n".join(queue_lines) + "\n").encode("utf-8"),
    )
    build_receipt = digested(
        {
            "schema": BUILD_RECEIPT_V2_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "base_package_id": PACKAGE_ID,
            "status": "passed_inert_collision_blocked",
            "baseline_full_validation": baseline_validation,
            "baseline_package_manifest": _json_binding(
                PACKAGE_DIR / PACKAGE_MANIFEST_NAME
            ),
            "immutable_compact_resume_inputs": {
                **_json_binding(
                    PACKAGE_DIR / RESUME_INPUTS_NAME
                ),
                "archive_count": RESUME_COUNT,
                "full_hash_and_member_scan_passed": True,
                "reused_without_copy_or_mutation": True,
            },
            "effective_sources_sha256": effective_sources["sha256"],
            "effective_source_family_count": 2,
            "changed_source_members": [CHECKPOINT_MEMBER],
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "repaired_checkpoint_sha256": (
                REPAIRED_CHECKPOINT_SHA256
            ),
            "protocol_members_byte_identical": True,
            "scientific_settings_changed": [],
            "derived_protocol_count": CELL_COUNT,
            "effective_execution_contract_count": CELL_COUNT,
            "settings_hash_removed": True,
            "separate_scientific_and_operational_hashes": True,
            "operator_sequence_match_claim_count": 0,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(
        PACKAGE_DIR / BUILD_RECEIPT_V2_NAME,
        build_receipt,
    )
    control_bindings = [
        _file_binding(PACKAGE_DIR / name)
        for name in OVERLAY_CONTROL_FILES
    ]
    manifest = digested(
        {
            "schema": OVERLAY_MANIFEST_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "base_package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "status": "passed_inert_collision_blocked",
            "cell_count": CELL_COUNT,
            "authenticated_resume_count": RESUME_COUNT,
            "fresh_count": FRESH_COUNT,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "base_package_manifest": _json_binding(
                PACKAGE_DIR / PACKAGE_MANIFEST_NAME
            ),
            "immutable_resume_inputs": {
                **_json_binding(
                    PACKAGE_DIR / RESUME_INPUTS_NAME
                ),
                "full_hash_and_member_scan_passed": True,
                "reused_without_copy_or_mutation": True,
            },
            "effective_sources": _json_binding(
                PACKAGE_DIR / EFFECTIVE_SOURCES_NAME
            ),
            "collision_evidence": _json_binding(
                PACKAGE_DIR / COLLISION_EVIDENCE_NAME
            ),
            "source_lock_audit": _json_binding(
                PACKAGE_DIR / SOURCE_LOCK_AUDIT_V2_NAME
            ),
            "execution_plan": _json_binding(
                PACKAGE_DIR / EXECUTION_PLAN_V2_NAME
            ),
            "queue": {
                **_file_binding(PACKAGE_DIR / QUEUE_V2_NAME),
                "row_count": CELL_COUNT,
                "column_count": 8,
                "kind": "inert_planning_queue_not_condor_queue",
            },
            "build_receipt": _json_binding(
                PACKAGE_DIR / BUILD_RECEIPT_V2_NAME
            ),
            "control_files": control_bindings,
            "jobs": job_bindings,
            "submit_descriptor_present": False,
            "authority_overlay_present": False,
            "collision_clearance_overlay_present": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / OVERLAY_MANIFEST_NAME, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "overlay_id": OVERLAY_ID,
                "package_id": OVERLAY_PACKAGE_ID,
                "cell_count": CELL_COUNT,
                "authenticated_resume_count": RESUME_COUNT,
                "fresh_count": FRESH_COUNT,
                "overlay_manifest_sha256": manifest["sha256"],
                "immutable_compact_resume_inputs": True,
                "submission_ready": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
