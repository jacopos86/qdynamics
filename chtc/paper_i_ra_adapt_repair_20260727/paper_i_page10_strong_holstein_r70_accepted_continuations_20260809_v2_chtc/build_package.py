#!/usr/bin/env python3
"""Seal and activate the three Page-10 strong-Holstein continuations."""

from __future__ import annotations

import gzip
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
    RECOVERABLE_PREFIX_MANIFEST_RELATIVE,
    RECOVERABLE_PREFIX_MANIFEST_SHA256,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REMOTE_OUTPUT_ROOT,
    RESOURCE_ENVELOPE,
    RESOURCE_WEIGHTING_SCOPE,
    ROUTE_CONTRACT_SHA256,
    ROUTE_ID,
    RUN_CLASS,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TARGET_ROUTE_PROFILE,
    VISIBLE_ADAPTER_FILE_SHA256,
    VISIBLE_ADAPTER_RELATIVE,
    VISIBLE_ADAPTER_SHA256,
    V1_CONTINUATION_MANIFEST_FILE_SHA256,
    V1_CONTINUATION_MANIFEST_SHA256,
    VENDORED_STREAMING_JSON_BACKEND,
    VENDORED_STREAMING_JSON_FILES,
    VENDORED_STREAMING_JSON_VERSION,
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
GENERATED_ROOTS = (
    "bundle",
    "source",
    "source_overlay",
    "resume_inputs",
    "jobs",
    "visible_source_resolution",
    "activation",
    "execution_plan.json",
    "queue.tsv",
    "package_manifest.json",
    "submit.sub",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o755 if executable else 0o644)


def _verify_file(path: Path, *, sha256: str, size_bytes: int | None = None) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or (size_bytes is not None and path.stat().st_size != size_bytes)
        or sha256_file(path) != sha256
    ):
        raise PackageContractError(f"Source binding drifted: {path}")


def _copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise PackageContractError(f"Unsafe copy source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)
    if sha256_file(source) != sha256_file(destination):
        raise PackageContractError(f"Exact copy drifted: {destination}")


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


def _build_local_resume_archive(spec: Mapping[str, Any], destination: Path) -> None:
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as archive:
                for row in spec["source_members"]:
                    source_path = REPO_ROOT / str(row["source_path"])
                    _verify_file(
                        source_path,
                        sha256=str(row["sha256"]),
                        size_bytes=int(row["size_bytes"]),
                    )
                    with source_path.open("rb") as source:
                        _tar_add_stream(
                            archive,
                            name=str(row["archive_path"]),
                            size=int(row["size_bytes"]),
                            source=source,
                        )


def _build_attempt_resume_archive(spec: Mapping[str, Any], destination: Path) -> None:
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
    observed: set[str] = set()
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(mode="w", fileobj=gz, format=tarfile.PAX_FORMAT) as output:
                with tarfile.open(source_path, "r:gz") as source_archive:
                    for member in source_archive:
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
                                f"Unsafe source resume member: {member.name}"
                            )
                        source = source_archive.extractfile(member)
                        if source is None:
                            raise PackageContractError(
                                f"Unreadable source resume member: {member.name}"
                            )
                        _tar_add_stream(
                            output,
                            name=str(row["archive_path"]),
                            size=int(row["size_bytes"]),
                            source=source,
                        )
                        observed.add(member.name)
    if observed != set(by_source):
        raise PackageContractError("Attempt resume source is incomplete.")


def _materialize_resume(spec: Mapping[str, Any]) -> dict[str, Any]:
    regime = str(spec["regime_id"])
    archive_path = PACKAGE_DIR / "resume_inputs" / f"{regime}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    source_validation = spec["v1_checkpoint_validation"]
    source_resume_binding = source_validation["source_resume_manifest"]
    source_resume_path = REPO_ROOT / str(source_resume_binding["path"])
    source_archive_path = (
        source_resume_path.parent
        / Path(str(source_validation["archive"]["path"])).name
    )
    _verify_file(
        source_archive_path,
        sha256=str(source_validation["archive"]["sha256"]),
        size_bytes=int(source_validation["archive"]["size_bytes"]),
    )
    _copy_exact(source_archive_path, archive_path)
    members = [
        {
            "role": row["role"],
            "path": row["archive_path"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            **(
                {"source_path": row["source_path"]}
                if "source_path" in row
                else {"source_member": row["source_member"]}
            ),
        }
        for row in spec["source_members"]
    ]
    archive_binding = file_binding(archive_path, root=PACKAGE_DIR)
    source_package_binding = source_validation["source_package_manifest"]
    source_package_path = REPO_ROOT / str(source_package_binding["path"])
    _verify_file(
        source_package_path,
        sha256=V1_CONTINUATION_MANIFEST_FILE_SHA256,
    )
    source_package = load_json(
        source_package_path, label="v1 continuation package manifest"
    )
    if (
        verify_self_digest(
            source_package, label="v1 continuation package manifest"
        )
        != V1_CONTINUATION_MANIFEST_SHA256
        or source_package_binding.get("canonical_sha256")
        != V1_CONTINUATION_MANIFEST_SHA256
    ):
        raise PackageContractError("V1 continuation package authority drifted.")
    _verify_file(
        source_resume_path,
        sha256=str(source_resume_binding["sha256"]),
    )
    source_resume = load_json(
        source_resume_path, label=f"{regime} v1 resume manifest"
    )
    if (
        verify_self_digest(
            source_resume, label=f"{regime} v1 resume manifest"
        )
        != source_resume_binding.get("canonical_sha256")
        or source_resume.get("archive") != archive_binding
        or source_resume.get("members") != members
        or source_validation.get("archive") != archive_binding
    ):
        raise PackageContractError(
            f"{regime} inherited validation bytes drifted."
        )
    checkpoint_validation = digested(
        {
            "schema": "paper_i_page10_checkpoint_validation_receipt_v1",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "regime_id": regime,
            "resume_round": int(spec["resume_round"]),
            "target_round": TARGET_HORIZON,
            "validation_authority": (
                "inherited_v1_full_stream_validation_exact_bytes_v1"
            ),
            "source_validation": source_validation,
            "archive": archive_binding,
            "member_count": 3,
            "members": members,
            "checkpoint_sha256": next(
                row["sha256"]
                for row in members
                if row["role"] == "checkpoint"
            ),
            "metadata": source_validation["metadata"],
            "worker_validation_scope": (
                "stream_authenticate_all_three_members_then_"
                "strict_resume_replay_v1"
            ),
            "accepted_state_resume_semantic_replay_required": True,
            "ambient_ijson_required": False,
        }
    )
    checkpoint_validation_path = (
        PACKAGE_DIR
        / "resume_inputs"
        / f"{regime}.checkpoint_validation.json"
    )
    _write_json(checkpoint_validation_path, checkpoint_validation)
    manifest = digested(
        {
            "schema": "paper_i_page10_pointer_closed_resume_archive_v1",
            "status": "passed",
            "package_id": PACKAGE_ID,
            "regime_id": regime,
            "resume_round": int(spec["resume_round"]),
            "target_round": TARGET_HORIZON,
            "source_kind": spec["source_kind"],
            "materialization_kind": (
                "exact_copy_authenticated_v1_compact_archive_v1"
            ),
            "member_count": 3,
            "members": members,
            "pointer_closed": True,
            "checkpoint_sha256": next(
                row["sha256"] for row in members if row["role"] == "checkpoint"
            ),
            "archive": archive_binding,
            "checkpoint_validation": file_binding(
                checkpoint_validation_path,
                root=PACKAGE_DIR,
                canonical=True,
            ),
        }
    )
    manifest_path = PACKAGE_DIR / "resume_inputs" / f"{regime}.manifest.json"
    _write_json(manifest_path, manifest)
    validated = validate_resume_archive(
        archive_path,
        manifest,
        expected_round=int(spec["resume_round"]),
        checkpoint_validation=checkpoint_validation,
    )
    if (
        int(
            validated["metadata"]["estimator_call_ledger_checkpoint"][
                "S_alg"
            ]
        )
        != int(spec["source_s_alg"])
    ):
        raise PackageContractError("Resume S_alg is invalid.")
    return manifest


def _base_protocol_path(regime: str) -> Path:
    return REPO_ROOT / BASE_PROTOCOL_ROOT / f"{source_execution_id(regime)}.json"


def _base_job_path(regime: str) -> Path:
    return BASE_PACKAGE / "jobs" / f"{source_execution_id(regime)}.json"


def _build_visible_source_map(adapter: Mapping[str, Any]) -> dict[str, Any]:
    adapter_cells = {
        str(row["regime_id"]): row
        for row in adapter.get("cells", [])
        if isinstance(row, Mapping)
    }
    regimes: dict[str, Any] = {}
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        cell = adapter_cells.get(regime)
        route = cell.get("macro_then_singleton") if isinstance(cell, Mapping) else None
        if (
            not isinstance(route, Mapping)
            or route.get("status") != "complete"
            or route.get("execution_id") != source_execution_id(regime)
        ):
            raise PackageContractError(f"Visible Page-10 cell drifted: {regime}")
        protocol_path = _base_protocol_path(regime)
        job_path = _base_job_path(regime)
        protocol = load_json(protocol_path, label=f"{regime} source protocol")
        job = load_json(job_path, label=f"{regime} source job")
        verify_self_digest(protocol, label=f"{regime} source protocol")
        verify_self_digest(job, label=f"{regime} source job")
        if (
            protocol.get("sha256") != spec["source_protocol_sha256"]
            or sha256_file(job_path) != spec["source_job_file_sha256"]
            or protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or int(protocol.get("horizon", -1)) != SOURCE_HORIZON
        ):
            raise PackageContractError(f"Page-10 source identity drifted: {regime}")
        regimes[regime] = {
            "nph": 7,
            "methods": {
                "macro_then_singleton": {
                    "visible_value": route.get("terminal", {}).get("error"),
                    "source_json": protocol_path.relative_to(REPO_ROOT).as_posix(),
                    "source_sha256": sha256_file(protocol_path),
                    "source_protocol_canonical_sha256": protocol["sha256"],
                    "source_job": {
                        "path": job_path.relative_to(REPO_ROOT).as_posix(),
                        "sha256": sha256_file(job_path),
                        "canonical_sha256": job["sha256"],
                    },
                    "execution_id": source_execution_id(regime),
                    "settings_changed": [
                        {
                            "path": "request.execution.stop.maximum_controller_rounds",
                            "before": SOURCE_HORIZON,
                            "after": TARGET_HORIZON,
                        }
                    ],
                }
            },
        }
    return digested(
        {
            "schema": "paper_i_page10_strong_r70_visible_source_map_v1",
            "figure_label": "Page 10",
            "source_adapter": {
                "path": VISIBLE_ADAPTER_RELATIVE.as_posix(),
                "sha256": VISIBLE_ADAPTER_FILE_SHA256,
                "canonical_sha256": VISIBLE_ADAPTER_SHA256,
            },
            "regimes": regimes,
        }
    )


def _derive_protocols(bundle_manifest_path: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        target = PACKAGE_DIR / "bundle" / "protocols" / f"{execution_id(regime)}.json"
        command = [
            sys.executable,
            "-B",
            (PACKAGE_DIR / "derive_protocol.py").as_posix(),
            "--base-package",
            BASE_PACKAGE.as_posix(),
            "--base-job",
            _base_job_path(regime).as_posix(),
            "--bundle-manifest",
            bundle_manifest_path.as_posix(),
            "--execution-id",
            execution_id(regime),
            "--regime-id",
            regime,
            "--target-horizon",
            str(TARGET_HORIZON),
            "--output",
            target.as_posix(),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise PackageContractError(
                f"Protocol derivation failed for {regime}: {completed.stderr}"
            )
        protocol = load_json(target, label=f"{regime} target protocol")
        verify_self_digest(protocol, label=f"{regime} target protocol")
        if (
            protocol.get("algorithm_id") != ALGORITHM_ID
            or protocol.get("horizon") != TARGET_HORIZON
            or protocol.get("route_contract", {}).get("sha256")
            != ROUTE_CONTRACT_SHA256
            or protocol.get("route_contract", {}).get("route_profile")
            != TARGET_ROUTE_PROFILE
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


def _expected_artifacts(execution: str) -> dict[str, Any]:
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


def _submit_descriptor(package_relative: str) -> str:
    vendored_runtime = ", ".join(
        f"{package_relative}/{relative}"
        for relative in VENDORED_STREAMING_JSON_FILES
    )
    return f"""universe = vanilla
executable = {package_relative}/execute_authorized_job.sh
transfer_executable = True

arguments = {package_relative} {package_relative}/$(job_path) {package_relative}/$(authorization_path) {package_relative}/source/source_locked.tar.gz {BASE_SOURCE_ARCHIVE_SHA256} {package_relative}/$(resume_archive) $(resume_archive_sha256) {REMOTE_IMAGE_PATH} {REMOTE_IMAGE_SHA256} $(execution_id) transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz

should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {package_relative}/package_contract.py, {package_relative}/run_cell.py, {vendored_runtime}, {package_relative}/package_manifest.json, {package_relative}/source/source_locked.tar.gz, {package_relative}/source/source_archive_manifest.json, {package_relative}/source/source_composition.json, {package_relative}/source_overlay/{CONTROLLER_RELATIVE_PATH}, {package_relative}/bundle/bundle_manifest.json, {package_relative}/bundle/source_locks.json, {package_relative}/$(job_path), {package_relative}/$(protocol_path), {package_relative}/$(resume_archive), {package_relative}/$(resume_manifest), {package_relative}/$(checkpoint_validation), {package_relative}/activation/activation_request.json, {package_relative}/$(authorization_path), {REMOTE_IMAGE_PATH}
transfer_output_files = transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz
transfer_output_remaps = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz={REMOTE_OUTPUT_ROOT}/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"

request_cpus = $(request_cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+MaxRuntime = $(max_runtime_seconds)
+JobBatchName = "paper-i-page10-strong-r70-cont-v2"

notification = Never
getenv = False
stream_output = False
stream_error = False
on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
periodic_release = False
leave_in_queue = False

log = {package_relative}_runtime/logs/$(Cluster).$(Process)-$(execution_id).log
output = {package_relative}_runtime/logs/$(Cluster).$(Process)-$(execution_id).out
error = {package_relative}_runtime/logs/$(Cluster).$(Process)-$(execution_id).err

queue execution_id, job_path, protocol_path, authorization_path, resume_archive, resume_manifest, checkpoint_validation, resume_archive_sha256, request_cpus, memory_mb, disk_mb, max_runtime_seconds from {package_relative}/queue.tsv
"""


def build() -> dict[str, Any]:
    if any((PACKAGE_DIR / path).exists() for path in GENERATED_ROOTS):
        raise FileExistsError("Refusing to overwrite a sealed continuation package.")
    for name in CONTROL_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise PackageContractError(f"Missing control file: {name}")

    base_manifest_path = BASE_PACKAGE / "package_manifest.json"
    _verify_file(base_manifest_path, sha256=BASE_PACKAGE_MANIFEST_FILE_SHA256)
    base_manifest = load_json(base_manifest_path, label="base package manifest")
    if verify_self_digest(base_manifest, label="base package manifest") != BASE_PACKAGE_MANIFEST_SHA256:
        raise PackageContractError("Base package canonical identity drifted.")
    base_source_path = BASE_PACKAGE / "source/source_locked.tar.gz"
    _verify_file(base_source_path, sha256=BASE_SOURCE_ARCHIVE_SHA256)
    base_source_manifest_path = BASE_PACKAGE / "source/source_archive_manifest.json"
    _verify_file(base_source_manifest_path, sha256=BASE_SOURCE_MANIFEST_FILE_SHA256)
    base_source_manifest = load_json(base_source_manifest_path, label="base source manifest")
    if verify_self_digest(base_source_manifest, label="base source manifest") != BASE_SOURCE_MANIFEST_SHA256:
        raise PackageContractError("Base source manifest identity drifted.")
    base_locks_path = (
        BASE_PACKAGE
        / "bundle_materialization"
        / "ra_adapt_macro_then_singleton_phase123_qiskit_phase23_no_lanes_tau1em4_r50_v1"
        / "source_locks.json"
    )
    _verify_file(base_locks_path, sha256=BASE_SOURCE_LOCKS_FILE_SHA256)
    base_locks = load_json(base_locks_path, label="base source locks")
    if verify_self_digest(base_locks, label="base source locks") != BASE_SOURCE_LOCKS_SHA256:
        raise PackageContractError("Base source-lock identity drifted.")
    adapter_path = REPO_ROOT / VISIBLE_ADAPTER_RELATIVE
    _verify_file(adapter_path, sha256=VISIBLE_ADAPTER_FILE_SHA256)
    adapter = load_json(adapter_path, label="Page-10 visible adapter")
    if verify_self_digest(adapter, label="Page-10 visible adapter") != VISIBLE_ADAPTER_SHA256:
        raise PackageContractError("Page-10 visible adapter identity drifted.")
    _verify_file(
        REPO_ROOT / RECOVERABLE_PREFIX_MANIFEST_RELATIVE,
        sha256=RECOVERABLE_PREFIX_MANIFEST_SHA256,
    )

    _copy_exact(base_source_path, PACKAGE_DIR / "source/source_locked.tar.gz")
    _copy_exact(
        base_source_manifest_path,
        PACKAGE_DIR / "source/source_archive_manifest.json",
    )
    _copy_exact(base_locks_path, PACKAGE_DIR / "bundle/source_locks.json")
    controller_source = REPO_ROOT / CONTROLLER_RELATIVE_PATH
    _verify_file(controller_source, sha256=CONTROLLER_AFTER_SHA256)
    controller_overlay = PACKAGE_DIR / "source_overlay" / CONTROLLER_RELATIVE_PATH
    _copy_exact(controller_source, controller_overlay)
    base_controller_rows = [
        row
        for row in base_source_manifest.get("members", [])
        if isinstance(row, Mapping) and row.get("path") == CONTROLLER_RELATIVE_PATH
    ]
    if len(base_controller_rows) != 1 or base_controller_rows[0].get("sha256") != CONTROLLER_BEFORE_SHA256:
        raise PackageContractError("Base controller source binding drifted.")
    regression_path = REPO_ROOT / "test/test_static_adapt_sr_snake_controller.py"
    source_composition = digested(
        {
            "schema": "paper_i_page10_r70_runtime_source_composition_v1",
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
            "operational_overlay": {
                "repair_id": CONTROLLER_REPAIR_ID,
                "path": CONTROLLER_RELATIVE_PATH,
                "before_sha256": CONTROLLER_BEFORE_SHA256,
                "after": file_binding(controller_overlay, root=PACKAGE_DIR),
                "semantic_scope": "accepted_energy_roundoff_only",
                "absolute_tolerance": "128*ulp(max(1,abs(E1),abs(E2)))",
                "all_non_energy_fields_exact": True,
                "scientific_protocol_changed": False,
                "scientific_settings_changed": [],
                "regression": {
                    "nodeid": CONTROLLER_REGRESSION,
                    "path": regression_path.relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256_file(regression_path),
                },
            },
            "streaming_json_runtime": {
                "distribution": "ijson",
                "version": VENDORED_STREAMING_JSON_VERSION,
                "backend": VENDORED_STREAMING_JSON_BACKEND,
                "implementation": "pure_python_source_locked_v1",
                "ambient_dependency_allowed": False,
                "license": file_binding(
                    PACKAGE_DIR / "vendor/ijson_pure/LICENSE.txt",
                    root=PACKAGE_DIR,
                ),
                "files": [
                    file_binding(PACKAGE_DIR / relative, root=PACKAGE_DIR)
                    for relative in VENDORED_STREAMING_JSON_FILES
                ],
            },
            "no_ambient_repo_imports": True,
        }
    )
    _write_json(PACKAGE_DIR / "source/source_composition.json", source_composition)

    source_map = _build_visible_source_map(adapter)
    source_map_path = PACKAGE_DIR / "visible_source_resolution/source_map.json"
    _write_json(source_map_path, source_map)
    resolver_bindings: list[dict[str, Any]] = []
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        trace = PACKAGE_DIR / "visible_source_resolution" / f"{regime}.resolver.json"
        completed = subprocess.run(
            [
                sys.executable,
                "agent_guidance/skills/shared/scripts/resolve_visible_settings.py",
                "--source-map",
                source_map_path.relative_to(REPO_ROOT).as_posix(),
                "--target-axis",
                "regimes",
                "--regime",
                regime,
                "--method",
                "macro_then_singleton",
                "--output-json",
                trace.relative_to(REPO_ROOT).as_posix(),
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise PackageContractError(
                f"Visible settings resolution failed for {regime}: {completed.stderr}"
            )
        resolver_bindings.append(
            {"regime_id": regime, **file_binding(trace, root=PACKAGE_DIR)}
        )

    bundle_manifest = digested(
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
                    "source_execution_id": source_execution_id(str(spec["regime_id"])),
                    "regime_id": spec["regime_id"],
                    "nph": 7,
                    "source_horizon": SOURCE_HORIZON,
                    "target_horizon": TARGET_HORIZON,
                    "resume_round": spec["resume_round"],
                }
                for spec in CELL_SPECS
            ],
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "source_locks_sha256": BASE_SOURCE_LOCKS_SHA256,
            "runtime_source_composition_sha256": source_composition["sha256"],
            "only_scientific_change": {
                "path": "request.execution.stop.maximum_controller_rounds",
                "before": SOURCE_HORIZON,
                "after": TARGET_HORIZON,
            },
            "execution_authorized": False,
            "submitted": False,
        }
    )
    bundle_manifest_path = PACKAGE_DIR / "bundle/bundle_manifest.json"
    _write_json(bundle_manifest_path, bundle_manifest)
    protocol_bindings = _derive_protocols(bundle_manifest_path)
    protocols_by_execution = {
        row["execution_id"]: row for row in protocol_bindings
    }
    resume_manifests = {
        str(spec["regime_id"]): _materialize_resume(spec) for spec in CELL_SPECS
    }

    jobs: list[dict[str, Any]] = []
    for spec in CELL_SPECS:
        regime = str(spec["regime_id"])
        execution = execution_id(regime)
        protocol_binding = protocols_by_execution[execution]
        protocol = load_json(
            PACKAGE_DIR / protocol_binding["path"], label=f"{regime} protocol"
        )
        resume_manifest = resume_manifests[regime]
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
                "resume_round": spec["resume_round"],
                "source_protocol_sha256": spec["source_protocol_sha256"],
                "protocol": protocol_binding,
                "protocol_sha256": protocol["sha256"],
                "resume_archive": resume_manifest["archive"],
                "resume_manifest": file_binding(
                    PACKAGE_DIR / "resume_inputs" / f"{regime}.manifest.json",
                    root=PACKAGE_DIR,
                    canonical=True,
                ),
                "checkpoint_validation": resume_manifest[
                    "checkpoint_validation"
                ],
                "checkpoint_sha256": resume_manifest["checkpoint_sha256"],
                "source_s_alg": spec["source_s_alg"],
                "runtime_source_composition_sha256": source_composition["sha256"],
                "resources": dict(RESOURCE_ENVELOPE),
                "expected_artifacts": _expected_artifacts(execution),
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        target = PACKAGE_DIR / "jobs" / f"{execution}.json"
        _write_json(target, job)
        jobs.append(job)

    queue_lines = []
    for job in jobs:
        resources = job["resources"]
        queue_lines.append(
            "\t".join(
                (
                    str(job["execution_id"]),
                    f"jobs/{job['execution_id']}.json",
                    str(job["protocol"]["path"]),
                    f"activation/authorizations/{job['execution_id']}.json",
                    str(job["resume_archive"]["path"]),
                    str(job["resume_manifest"]["path"]),
                    str(job["checkpoint_validation"]["path"]),
                    str(job["resume_archive"]["sha256"]),
                    str(resources["request_cpus"]),
                    str(resources["request_memory_mb"]),
                    str(resources["request_disk_mb"]),
                    str(resources["max_runtime_seconds"]),
                )
            )
        )
    _write_bytes(PACKAGE_DIR / "queue.tsv", ("\n".join(queue_lines) + "\n").encode())
    plan = digested(
        {
            "schema": "paper_i_page10_strong_r70_continuation_plan_v1",
            "status": "passed_inert",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "row_count": 3,
            "execution_ids": list(expected_execution_ids()),
            "route_contract_sha256": ROUTE_CONTRACT_SHA256,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "resume_rounds": {
                str(spec["regime_id"]): spec["resume_round"] for spec in CELL_SPECS
            },
            "resources": dict(RESOURCE_ENVELOPE),
            "runtime_source_composition_sha256": source_composition["sha256"],
            "execution_authorized": False,
            "submitted": False,
        }
    )
    _write_json(PACKAGE_DIR / "execution_plan.json", plan)
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
            "visible_source_map": file_binding(
                source_map_path, root=PACKAGE_DIR, canonical=True
            ),
            "visible_source_resolver_traces": resolver_bindings,
            "runtime_source_composition": file_binding(
                PACKAGE_DIR / "source/source_composition.json",
                root=PACKAGE_DIR,
                canonical=True,
            ),
            "bundle_manifest": file_binding(
                bundle_manifest_path, root=PACKAGE_DIR, canonical=True
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
            "queue": file_binding(PACKAGE_DIR / "queue.tsv", root=PACKAGE_DIR),
            "execution_plan": file_binding(
                PACKAGE_DIR / "execution_plan.json", root=PACKAGE_DIR, canonical=True
            ),
            "control_files": [
                file_binding(PACKAGE_DIR / name, root=PACKAGE_DIR)
                for name in CONTROL_FILES
            ],
            "remote_image_path": REMOTE_IMAGE_PATH,
            "remote_image_sha256": REMOTE_IMAGE_SHA256,
            "remote_output_root": REMOTE_OUTPUT_ROOT,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / "package_manifest.json", manifest)

    activation_request = digested(
        {
            "schema": "paper_i_page10_strong_r70_activation_request_v1",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest_sha256": manifest["sha256"],
            "requested_execution_ids": list(expected_execution_ids()),
            "authority_scope": "three_strong_holstein_continuations_to_round_70",
            "authorization_kind": "standing_explicit_user_chtc_authority",
            "execution_authorized": True,
            "submission_authorized": True,
            "paper_evidence_adoption_authorized": False,
            "submitted": False,
        }
    )
    request_path = PACKAGE_DIR / "activation/activation_request.json"
    _write_json(request_path, activation_request)
    authorization_bindings: list[dict[str, Any]] = []
    for job in jobs:
        authority = digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "package_id": PACKAGE_ID,
                "campaign_id": CAMPAIGN_ID,
                "execution_id": job["execution_id"],
                "package_manifest_sha256": manifest["sha256"],
                "activation_request_sha256": activation_request["sha256"],
                "job_spec_sha256": job["sha256"],
                "protocol_sha256": job["protocol_sha256"],
                "resume_archive_sha256": job["resume_archive"]["sha256"],
                "checkpoint_sha256": job["checkpoint_sha256"],
                "checkpoint_validation_sha256": job[
                    "checkpoint_validation"
                ]["canonical_sha256"],
                "runtime_source_composition_sha256": source_composition["sha256"],
                "pinned_image_sha256": REMOTE_IMAGE_SHA256,
                "authorization_kind": "standing_explicit_user_chtc_authority",
                "execution_authorized": True,
                "submission_authorized": True,
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
        authorization_bindings.append(
            {
                "execution_id": job["execution_id"],
                **file_binding(auth_path, root=PACKAGE_DIR, canonical=True),
            }
        )

    package_relative = PACKAGE_DIR.relative_to(REPO_ROOT).as_posix()
    descriptor = _submit_descriptor(package_relative)
    _write_bytes(PACKAGE_DIR / "submit.sub", descriptor.encode())
    activation = digested(
        {
            "schema": ACTIVATION_SCHEMA,
            "status": "passed_activation_prepared_no_submission",
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "package_manifest_sha256": manifest["sha256"],
            "activation_request": file_binding(
                request_path, root=PACKAGE_DIR, canonical=True
            ),
            "authorizations": authorization_bindings,
            "authorization_count": 3,
            "submit_descriptor": file_binding(
                PACKAGE_DIR / "submit.sub", root=PACKAGE_DIR
            ),
            "remote_output_root": REMOTE_OUTPUT_ROOT,
            "explicit_transfer_output_files": True,
            "unique_posix_staging_remaps": True,
            "row_sharded_resume_transfers": True,
            "execution_authorized": True,
            "submission_authorized": True,
            "launch_ready": True,
            "submitted": False,
            "remote_stage": False,
            "condor_submit": False,
        }
    )
    _write_json(PACKAGE_DIR / "activation/activation_manifest.json", activation)
    return {
        "status": "passed_activation_prepared_no_submission",
        "package_id": PACKAGE_ID,
        "package_manifest_sha256": manifest["sha256"],
        "activation_manifest_sha256": activation["sha256"],
        "route_contract_sha256": ROUTE_CONTRACT_SHA256,
        "row_count": 3,
        "submitted": False,
    }


if __name__ == "__main__":
    try:
        print(canonical_json_bytes(build()).decode("utf-8"))
    except (FileExistsError, OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
