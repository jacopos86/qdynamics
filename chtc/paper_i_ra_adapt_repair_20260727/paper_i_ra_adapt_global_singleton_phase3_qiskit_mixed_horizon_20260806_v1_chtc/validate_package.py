#!/usr/bin/env python3
"""Validate the inert Phase-III-only Qiskit CHTC package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from package_contract import (  # noqa: E402
    ALGORITHM_ID,
    BACKEND_COMPILE_SCOPE,
    BUNDLE_ID,
    CAMPAIGN_ID,
    CONTROL_FILES,
    EXECUTION_PLAN_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_SCHEMA,
    REGIME_ROWS,
    REMOTE_IMAGE_PATH,
    REMOTE_IMAGE_SHA256,
    REQUIRED_PHASE3_QISKIT_SOURCE_PATHS,
    SOURCE_ROUTE_CONTRACT_SHA256,
    SOURCE_ROUTE_PROFILE,
    TARGET_ROUTE_PROFILE,
    PackageContractError,
    canonical_json_bytes,
    digested,
    expected_execution_ids,
    load_json,
    safe_relative_path,
    sha256_file,
    source_lock_id,
    verify_self_digest,
)
from run_cell import _load_closed_job  # noqa: E402


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _package_path(value: Any, *, label: str) -> Path:
    path = PACKAGE_DIR / safe_relative_path(value, label=label)
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(f"{label} escaped package.") from exc
    return path


def _bound_file(
    raw: Any,
    *,
    label: str,
    canonical: bool = False,
) -> tuple[Path, dict[str, Any] | None]:
    row = _mapping(raw, label=f"{label} binding")
    path = _package_path(row.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(row.get("size_bytes", -1))
        or sha256_file(path) != row.get("sha256")
    ):
        raise PackageContractError(f"{label} byte binding drifted.")
    if not canonical:
        return path, None
    payload = load_json(path, label=label)
    if verify_self_digest(payload, label=label) != row.get(
        "canonical_sha256"
    ):
        raise PackageContractError(f"{label} canonical binding drifted.")
    return path, payload


def _verify_archive(
    archive_path: Path,
    rows: list[Any],
    *,
    label: str,
) -> set[str]:
    declared = {
        safe_relative_path(row.get("path"), label=f"{label} member").as_posix(): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows):
        raise PackageContractError(f"{label} member declaration drifted.")
    observed: set[str] = set()
    import hashlib

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            row = declared.get(member.name)
            if (
                row is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"Unsafe {label} member: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable {label} member: {member.name}"
                )
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != row.get("sha256"):
                raise PackageContractError(
                    f"{label} member hash drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(declared):
        raise PackageContractError(f"{label} member closure failed.")
    return observed


def _resolve_pinned_image_runtime(
    *,
    image_path: Path,
    container_runtime: str | None,
) -> Path:
    if (
        not image_path.is_file()
        or image_path.is_symlink()
        or sha256_file(image_path) != REMOTE_IMAGE_SHA256
    ):
        raise PackageContractError("Pinned execution image hash drifted.")
    runtime = container_runtime
    if runtime is None:
        runtime = shutil.which("apptainer") or shutil.which("singularity")
    if runtime is None:
        raise PackageContractError(
            "Pinned image was supplied but no Apptainer/Singularity runtime "
            "is available."
        )
    runtime_path = Path(runtime).expanduser().resolve()
    if not runtime_path.is_file() or runtime_path.is_symlink():
        raise PackageContractError("Container runtime is missing or unsafe.")
    return runtime_path


def _worker_preflight(
    job_path: Path,
    *,
    image_path: Path | None = None,
    container_runtime: Path | None = None,
) -> dict[str, Any]:
    if image_path is None:
        command = [
            sys.executable,
            "-B",
            str(PACKAGE_DIR / "run_cell.py"),
            "--preflight",
            "--job",
            str(job_path),
        ]
    else:
        if container_runtime is None:
            raise PackageContractError(
                "Container worker preflight requires a resolved runtime."
            )
        mount = "/phase3_qiskit_package"
        command = [
            str(container_runtime),
            "exec",
            "--cleanenv",
            "--bind",
            f"{PACKAGE_DIR}:{mount}:ro",
            str(image_path.resolve()),
            "python3",
            "-B",
            f"{mount}/run_cell.py",
            "--preflight",
            "--job",
            f"{mount}/jobs/{job_path.name}",
        ]
    completed = subprocess.run(
        command,
        cwd=PACKAGE_DIR,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "off",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            f"Worker preflight failed for {job_path.name}: "
            f"{completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PackageContractError("Worker preflight emitted stdout noise.")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PackageContractError(
            "Worker preflight receipt is not JSON."
        ) from exc
    if not isinstance(receipt, dict):
        raise PackageContractError("Worker preflight receipt is malformed.")
    verify_self_digest(receipt, label="worker preflight receipt")
    if (
        receipt.get("status") != "passed"
        or receipt.get("scientific_execution_performed") is not False
    ):
        raise PackageContractError("Worker preflight did not remain inert.")
    return receipt


def _probe_image_runtime(
    *,
    image_path: Path,
    container_runtime: str | None,
) -> dict[str, Any]:
    runtime_path = _resolve_pinned_image_runtime(
        image_path=image_path,
        container_runtime=container_runtime,
    )
    mount = "/phase3_qiskit_package"
    completed = subprocess.run(
        [
            str(runtime_path),
            "exec",
            "--cleanenv",
            "--bind",
            f"{PACKAGE_DIR}:{mount}:ro",
            str(image_path.resolve()),
            "python3",
            "-B",
            f"{mount}/probe_image_runtime.py",
            "--source-archive",
            f"{mount}/source/source_locked.tar.gz",
        ],
        cwd=PACKAGE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PackageContractError(
            "Pinned image FakeMarrakesh probe failed: "
            + completed.stderr.strip()
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PackageContractError("Pinned image probe emitted stdout noise.")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PackageContractError("Pinned image probe is not JSON.") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "passed"
        or receipt.get("resolved_backend_name") != "FakeMarrakesh"
        or receipt.get("backend_resolution_kind") != "fake_exact"
        or receipt.get("optimization_level") != 1
        or receipt.get("seed_transpiler") != 7
        or receipt.get("structure_theta_value") != 1.0
        or receipt.get("allow_preferred_fallback") is not False
        or receipt.get("reward_negative_deltas") is not False
        or receipt.get("one_qubit_coordinate_policy")
        != "compiled_positive_delta_v1"
        or receipt.get("sealed_source_imported") is not True
        or int(receipt.get("compiled_depth", 0)) <= 0
    ):
        raise PackageContractError(
            "Pinned image oracle/transpile semantics drifted."
        )
    return {
        "status": "passed",
        "image_path": str(image_path.resolve()),
        "image_sha256": REMOTE_IMAGE_SHA256,
        "container_runtime": str(runtime_path),
        "probe": receipt,
    }


def _validate_matchmaking_safe_submit_template(template: str) -> None:
    submit_lines = [line.strip() for line in template.splitlines() if line.strip()]
    expected_resource_assignments = {
        "request_cpus = $(cpu_count)",
        "request_memory = $(memory_mib)MB",
        "request_disk = $(disk_mib)MB",
        "+MaxRuntime = $(runtime_seconds)",
    }
    expected_queue_line = (
        "queue execution_id, job_path, protocol_path, job_sha256, "
        "cpu_count, memory_mib, disk_mib, runtime_seconds from "
        "__PACKAGE_REL__/queue.tsv"
    )
    assignment_rows: list[tuple[str, str]] = []
    for line in submit_lines:
        if line.startswith("#") or "=" not in line:
            continue
        raw_name = line.split("=", 1)[0].strip()
        normalized_name = raw_name.lstrip("+").casefold()
        if normalized_name.startswith("my."):
            normalized_name = normalized_name[3:]
        assignment_rows.append((line, normalized_name))
    standard_resource_names = {
        "request_cpus",
        "request_memory",
        "request_disk",
        "maxruntime",
    }
    observed_resource_assignments = [
        line
        for line, name in assignment_rows
        if name in standard_resource_names
    ]
    forbidden_assignment = any(
        name == "requirements"
        or (name.startswith("request") and name not in standard_resource_names)
        for _line, name in assignment_rows
    )
    observed_queue_lines = [
        line for line in submit_lines if line.lower().startswith("queue ")
    ]
    if (
        len(observed_resource_assignments) != 4
        or set(observed_resource_assignments) != expected_resource_assignments
        or observed_queue_lines != [expected_queue_line]
        or forbidden_assignment
        or "$(request_" in template.lower()
        or "$(max_runtime_seconds)" in template
    ):
        raise PackageContractError(
            "Matchmaking-safe submit-variable contract drifted."
        )


def validate_package(
    *,
    deep: bool = False,
    image_path: Path | None = None,
    container_runtime: str | None = None,
    require_launch_ready: bool = False,
) -> dict[str, Any]:
    forbidden = [
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    if forbidden:
        raise PackageContractError(f"Unbound bytecode present: {forbidden}")
    if (PACKAGE_DIR / "authorizations").exists():
        raise PackageContractError("Inert package must not contain authorizations.")
    if (PACKAGE_DIR / "submit.sub").exists():
        raise PackageContractError("Inert package must not contain submit.sub.")

    manifest = load_json(
        PACKAGE_DIR / "package_manifest.json",
        label="package manifest",
    )
    verify_self_digest(manifest, label="package manifest")
    expected_ids = list(expected_execution_ids())
    if (
        manifest.get("schema") != PACKAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "passed_inert_six_cells"
        or manifest.get("package_id") != PACKAGE_ID
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("bundle_id") != BUNDLE_ID
        or manifest.get("row_count") != 6
        or manifest.get("execution_ids") != expected_ids
        or manifest.get("weak_holstein_horizon") != 50
        or manifest.get("strong_holstein_horizon") != 70
        or manifest.get("source_route_contract_sha256")
        != SOURCE_ROUTE_CONTRACT_SHA256
        or manifest.get("remote_image_path") != REMOTE_IMAGE_PATH
        or manifest.get("remote_image_sha256") != REMOTE_IMAGE_SHA256
        or manifest.get("remote_image_runtime_probe_required") is not True
        or manifest.get("activation_policy")
        != "fresh_explicit_user_authority_plus_pinned_image_probe_v1"
        or manifest.get("activation_artifacts_present") is not False
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submission_ready") is not False
        or manifest.get("submit_template_present") is not True
        or manifest.get("submit_descriptor_present") is not False
        or manifest.get("submitted") is not False
        or manifest.get("remote_stage") is not False
        or manifest.get("condor_submit") is not False
    ):
        raise PackageContractError("Package manifest semantic closure drifted.")
    child_route_sha = str(manifest.get("child_route_contract_sha256", ""))
    if len(child_route_sha) != 64 or child_route_sha == SOURCE_ROUTE_CONTRACT_SHA256:
        raise PackageContractError("Child route digest is missing or stale.")

    controls = _sequence(manifest.get("control_files"), label="control files")
    if [row.get("path") for row in controls if isinstance(row, Mapping)] != list(
        CONTROL_FILES
    ):
        raise PackageContractError("Control-file order or closure drifted.")
    for row in controls:
        control_path, _ = _bound_file(
            row,
            label=f"control {row.get('path')}",
        )
        if row.get("path") == "execute_authorized_job.sh" and not os.access(
            control_path, os.X_OK
        ):
            raise PackageContractError("Execution wrapper is not executable.")
    wrapper_text = (PACKAGE_DIR / "execute_authorized_job.sh").read_text(
        encoding="utf-8"
    )
    submit_template_text = (PACKAGE_DIR / "submit.sub.in").read_text(
        encoding="utf-8"
    )
    if (
        f'pinned_image_sha256="{REMOTE_IMAGE_SHA256}"' not in wrapper_text
        or f" {REMOTE_IMAGE_SHA256} " not in submit_template_text
        or "__ACTIVATION_REL__" not in submit_template_text
        or "__ACTIVATION_REL__," not in submit_template_text
    ):
        raise PackageContractError("Pinned-image activation controls drifted.")
    _validate_matchmaking_safe_submit_template(submit_template_text)

    _authority_path, authority = _bound_file(
        manifest.get("source_authority_manifest"),
        label="source authority manifest",
        canonical=True,
    )
    authority_archive, _ = _bound_file(
        manifest.get("source_authority_archive"),
        label="source authority archive",
    )
    _derivation_path, derivation = _bound_file(
        manifest.get("source_lock_derivation_receipt"),
        label="source-lock derivation receipt",
        canonical=True,
    )
    _bundle_path, bundle = _bound_file(
        manifest.get("bundle_manifest"),
        label="bundle manifest",
        canonical=True,
    )
    _locks_path, locks = _bound_file(
        manifest.get("bundle_source_locks"),
        label="bundle source locks",
        canonical=True,
    )
    _validation_path, bundle_validation = _bound_file(
        manifest.get("bundle_validation_report"),
        label="bundle validation report",
        canonical=True,
    )
    _plan_path, plan = _bound_file(
        manifest.get("execution_plan"),
        label="execution plan",
        canonical=True,
    )
    _audit_path, audit = _bound_file(
        manifest.get("source_lock_audit"),
        label="source-lock audit",
        canonical=True,
    )
    assert all(
        value is not None
        for value in (
            authority,
            derivation,
            bundle,
            locks,
            bundle_validation,
            plan,
            audit,
        )
    )
    assert authority is not None
    assert derivation is not None
    assert bundle is not None
    assert locks is not None
    assert bundle_validation is not None
    assert plan is not None
    assert audit is not None
    if (
        authority.get("source_route_contract_sha256")
        != SOURCE_ROUTE_CONTRACT_SHA256
        or authority.get("protocol_count") != 6
        or authority.get("execution_authorized") is not False
        or derivation.get("source_authority_manifest_sha256")
        != authority.get("sha256")
        or derivation.get("cell_count") != 6
        or derivation.get("resolver_trace_reconstruction_policy")
        != (
            "exact_page7_parent_protocol_only_no_historical_"
            "archive_authority_v1"
        )
        or derivation.get("stale_cluster_9381198_authority_retained")
        is not False
        or derivation.get("stale_core_fixed_horizon_delta_retained")
        is not False
        or bundle.get("campaign_id") != CAMPAIGN_ID
        or bundle.get("bundle_id") != BUNDLE_ID
        or bundle.get("execution_authorized") is not False
        or locks.get("all_required_files_verified") is not True
        or locks.get("required_cell_lock_count") != 6
        or bundle_validation.get("materialization_status") != "passed"
        or plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("status") != "passed_inert_six_cells"
        or plan.get("child_route_contract_sha256") != child_route_sha
        or plan.get("activation_policy")
        != "fresh_explicit_user_authority_plus_pinned_image_probe_v1"
        or plan.get("activation_artifacts_present") is not False
        or plan.get("execution_authorized") is not False
        or plan.get("submission_ready") is not False
        or audit.get("status") != "passed"
        or audit.get("source_route_profile") != SOURCE_ROUTE_PROFILE
        or audit.get("source_route_contract_sha256")
        != SOURCE_ROUTE_CONTRACT_SHA256
        or audit.get("child_route_profile") != TARGET_ROUTE_PROFILE
        or audit.get("child_route_contract_sha256") != child_route_sha
    ):
        raise PackageContractError("Authority/bundle/plan closure drifted.")

    authority_protocols = _sequence(
        authority.get("protocols"), label="source authority protocols"
    )
    with tarfile.open(authority_archive, "r:gz") as archive:
        authority_member_sizes = {
            member.name: member.size for member in archive.getmembers()
        }
    authority_rows = [
        {
            "path": row["member_path"],
            "sha256": row["file_sha256"],
            "size_bytes": authority_member_sizes.get(row["member_path"]),
        }
        for row in authority_protocols
        if isinstance(row, Mapping)
    ]
    if len(authority_rows) != 6:
        raise PackageContractError("Source authority protocol closure drifted.")
    _verify_archive(
        authority_archive,
        authority_rows,
        label="source authority archive",
    )

    source_archive, _ = _bound_file(
        manifest.get("source_archive"), label="runtime source archive"
    )
    _source_manifest_path, source_manifest = _bound_file(
        manifest.get("source_archive_manifest"),
        label="runtime source archive manifest",
        canonical=True,
    )
    assert source_manifest is not None
    source_rows = _sequence(
        source_manifest.get("members"), label="runtime source members"
    )
    source_members = _verify_archive(
        source_archive,
        source_rows,
        label="runtime source archive",
    )
    missing_route_sources = sorted(
        set(REQUIRED_PHASE3_QISKIT_SOURCE_PATHS).difference(source_members)
    )
    if missing_route_sources or manifest.get(
        "required_route_source_paths"
    ) != list(REQUIRED_PHASE3_QISKIT_SOURCE_PATHS):
        raise PackageContractError(
            "Runtime source archive omitted Phase-III-Qiskit source: "
            + ", ".join(missing_route_sources)
        )
    implementation_rows = locks.get("implementation_sources", {}).get("files")
    if not isinstance(implementation_rows, list):
        raise PackageContractError("Implementation source inventory is absent.")
    implementation_hashes = {
        str(row.get("path")): str(row.get("sha256"))
        for row in implementation_rows
        if isinstance(row, Mapping)
    }
    archive_hashes = {
        str(row.get("path")): str(row.get("sha256"))
        for row in source_rows
        if isinstance(row, Mapping)
    }
    for relative in REQUIRED_PHASE3_QISKIT_SOURCE_PATHS:
        if implementation_hashes.get(relative) != archive_hashes.get(relative):
            raise PackageContractError(
                f"Route source hash is not sealed exactly: {relative}"
            )

    cell_locks = _mapping(locks.get("cell_locks"), label="cell source locks")
    expected_delta_ids = {
        "phase3_qiskit_selector_cost_scope",
        "phase3_qiskit_exact_cell_selection",
    }
    for regime_id, nph, horizon in REGIME_ROWS:
        lock_id = source_lock_id(regime_id, nph)
        lock = _mapping(cell_locks.get(lock_id), label=f"source lock {lock_id}")
        trace = _mapping(
            lock.get("resolver_trace"), label=f"resolver trace {lock_id}"
        )
        changes = _sequence(
            trace.get("settings_changed"), label=f"settings changes {lock_id}"
        )
        change_ids = {
            str(change.get("id"))
            for change in changes
            if isinstance(change, Mapping)
        }
        settings_authority = _mapping(
            trace.get("settings_reused_sources"),
            label=f"settings authority {lock_id}",
        )
        page7_authority = _mapping(
            trace.get("page7_parent_protocol_authority"),
            label=f"page-7 authority {lock_id}",
        )
        phase3_anchor = _mapping(
            trace.get("phase3_qiskit_source_anchor"),
            label=f"Phase-III-Qiskit anchor {lock_id}",
        )
        trace_bytes = canonical_json_bytes(trace)
        if (
            b"9381198" in trace_bytes
            or b"all_six_r50" in trace_bytes
            or "global_singleton_source_anchor" in trace
            or "source_archive_member_authority" in trace
            or change_ids != expected_delta_ids
            or settings_authority.get("authority_kind")
            != "exact_page7_resolved_protocol_bytes_v1"
            or settings_authority.get("archive") != lock.get("archive")
            or settings_authority.get("archive_member") != lock.get("member")
            or page7_authority != settings_authority
            or int(phase3_anchor.get("source_horizon", -1)) != horizon
            or int(phase3_anchor.get("target_horizon", -1)) != horizon
            or phase3_anchor.get("scientific_result_anchor_claimed") is not False
        ):
            raise PackageContractError(
                f"Page-7-only resolver authority drifted: {lock_id}."
            )

    protocol_rows = _sequence(
        manifest.get("protocols"), label="protocol bindings"
    )
    if [row.get("execution_id") for row in protocol_rows if isinstance(row, Mapping)] != expected_ids:
        raise PackageContractError("Protocol order drifted.")
    jobs: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    worker_runtime: Path | None = None
    if deep and image_path is not None:
        worker_runtime = _resolve_pinned_image_runtime(
            image_path=image_path,
            container_runtime=container_runtime,
        )
    for execution_id, (regime_id, nph, horizon) in zip(expected_ids, REGIME_ROWS):
        job_path = PACKAGE_DIR / "jobs" / f"{execution_id}.json"
        job, _manifest, protocol, _source_locks = _load_closed_job(job_path)
        route = _mapping(protocol.get("route_contract"), label="route contract")
        execution = _mapping(
            route.get("execution_settings"), label="route execution"
        )
        invariants = _mapping(
            route.get("semantic_invariants"), label="route invariants"
        )
        lineage = _mapping(
            route.get("lineage_authority"), label="route lineage"
        )
        if (
            job.get("regime_id") != regime_id
            or job.get("nph") != nph
            or job.get("target_horizon") != horizon
            or protocol.get("horizon") != horizon
            or protocol.get("algorithm_id") != ALGORITHM_ID
            or route.get("route_profile") != TARGET_ROUTE_PROFILE
            or route.get("sha256") != child_route_sha
            or execution.get("phase3_backend_cost_scope")
            != BACKEND_COMPILE_SCOPE
            or execution.get("phase3_hardware_cost_normalization_mode")
            != "family_robust_symmetric_arctan_v1"
            or invariants.get(
                "phase_iii_qiskit_independent_base_trial_layouts"
            )
            is not True
            or invariants.get(
                "phase_iii_qiskit_population_normalization_policy"
            )
            != "family_robust_symmetric_arctan_v1"
            or lineage.get("parent_route_profile") != SOURCE_ROUTE_PROFILE
            or lineage.get("parent_contract_sha256")
            != SOURCE_ROUTE_CONTRACT_SHA256
        ):
            raise PackageContractError(f"Protocol drifted: {execution_id}")
        jobs.append(job)
        if deep:
            preflights.append(
                _worker_preflight(
                    job_path,
                    image_path=image_path,
                    container_runtime=worker_runtime,
                )
            )

    expected_queue = "".join(
        "\t".join(
            (
                job["execution_id"],
                job["job_path"],
                job["protocol_path"],
                job["sha256"],
                str(job["resources"]["request_cpus"]),
                str(job["resources"]["request_memory_mb"]),
                str(job["resources"]["request_disk_mb"]),
                str(job["resources"]["max_runtime_seconds"]),
            )
        )
        + "\n"
        for job in jobs
    )
    queue_path, _ = _bound_file(manifest.get("queue"), label="queue")
    if queue_path.read_text(encoding="utf-8") != expected_queue:
        raise PackageContractError("queue.tsv drifted.")

    if image_path is None:
        image_probe = {
            "status": "required_not_run",
            "pinned_path": REMOTE_IMAGE_PATH,
            "pinned_sha256": REMOTE_IMAGE_SHA256,
            "reason": "pinned_image_not_supplied_to_local_validator",
        }
        launch_ready = False
    else:
        image_probe = _probe_image_runtime(
            image_path=image_path,
            container_runtime=container_runtime,
        )
        launch_ready = True
    if require_launch_ready and not launch_ready:
        raise PackageContractError(
            "Launch readiness requires a passing pinned-image runtime probe."
        )

    return digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_phase3_qiskit_"
                "package_validation_v1"
            ),
            "status": "passed_inert_package",
            "package_id": PACKAGE_ID,
            "package_manifest_sha256": manifest["sha256"],
            "source_route_contract_sha256": SOURCE_ROUTE_CONTRACT_SHA256,
            "child_route_contract_sha256": child_route_sha,
            "cell_count": len(jobs),
            "deep_worker_preflight_count": len(preflights),
            "deep_worker_preflight_runtime": (
                "not_run"
                if not preflights
                else (
                    "pinned_execution_image"
                    if image_path is not None
                    else "host_interpreter"
                )
            ),
            "sealed_required_route_source_count": len(
                REQUIRED_PHASE3_QISKIT_SOURCE_PATHS
            ),
            "pinned_image_runtime_probe": image_probe,
            "launch_ready": launch_ready,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
            "writes_performed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--container-runtime")
    parser.add_argument("--require-launch-ready", action="store_true")
    args = parser.parse_args()
    try:
        receipt = validate_package(
            deep=args.deep,
            image_path=(None if args.image is None else args.image.resolve()),
            container_runtime=args.container_runtime,
            require_launch_ready=args.require_launch_ready,
        )
    except (OSError, PackageContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
