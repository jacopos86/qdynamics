#!/usr/bin/env python3
"""Materialize the inert 36-resume r50->r70 planning scaffold."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from scaffold_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    CAMPAIGN_ID,
    CELL_COUNT,
    CONTROLLED_CYCLE_VALIDATOR_BINDING,
    CONTROL_FILES,
    EXECUTION_MODE,
    EXECUTION_TARGET,
    GENERATED_FILES,
    INHERITED_RESUME_COUNT,
    JOBS_DIR,
    ONLY_SCIENTIFIC_CHANGE,
    PACKAGE_ID,
    PACKAGE_PLAN_NAME,
    PACKAGE_PLAN_SCHEMA,
    PACKAGE_RELATIVE_ROOT,
    PENDING_PREDECESSORS,
    PENDING_RESUME_COUNT,
    PREDECESSOR_REQUIREMENT_SCHEMA,
    PREDECESSOR_PLACEHOLDERS_DIR,
    PREDECESSOR_REQUIREMENTS_NAME,
    RESOURCE_WEIGHTING_SCOPE,
    RUN_CLASS,
    RUNTIME_BUNDLE_RELATIVE,
    SCAFFOLD_JOB_SCHEMA,
    SCAFFOLD_MANIFEST_NAME,
    SCAFFOLD_MANIFEST_SCHEMA,
    SEALED_PARENT_MANIFEST_CANONICAL_SHA256,
    SEALED_PARENT_MANIFEST_FILE_SHA256,
    SEALED_PARENT_MANIFEST_NAME,
    SEALED_PARENT_RELATIVE_ROOT,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TARGET_RESUME_COUNT,
    TRANSFER_PLAN_NAME,
    TRANSFER_PLAN_SCHEMA,
    TRANSFER_QUEUE_NAME,
    ScaffoldContractError,
    canonical_json_bytes,
    digested,
    load_json,
    load_sealed_parent_jobs,
    package_root,
    parent_root,
    predecessor_requirement,
    repo_file_binding,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    validate_scientific_projection,
    verify_exact_binding,
)


def _exclusive_write(
    path: Path, data: bytes, *, executable: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ScaffoldContractError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
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


def _canonical_copy(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _parent_job_bindings(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in manifest["jobs"]:
        if not isinstance(raw, Mapping):
            raise ScaffoldContractError("Parent job binding is malformed.")
        result[str(raw["execution_id"])] = _canonical_copy(raw)
    return result


def _repo_binding_from_parent_binding(
    *,
    repo_root: Path,
    relative_to_parent: Mapping[str, Any],
    label: str,
    rehash: bool,
) -> dict[str, Any]:
    relative = safe_relative_path(
        relative_to_parent.get("path"), label=f"{label} path"
    )
    path = parent_root(repo_root) / relative
    verify_exact_binding(
        path,
        relative_to_parent,
        label=label,
        rehash=rehash,
    )
    result = {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": str(relative_to_parent["sha256"]),
        "size_bytes": int(relative_to_parent["size_bytes"]),
    }
    if relative_to_parent.get("canonical_sha256") is not None:
        result["canonical_sha256"] = str(
            relative_to_parent["canonical_sha256"]
        )
    return result


def _resume_input_from_parent(
    *,
    repo_root: Path,
    parent_job: Mapping[str, Any],
    rehash: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resume = parent_job.get("resume_input")
    if not isinstance(resume, Mapping):
        raise ScaffoldContractError(
            f"{parent_job.get('execution_id')} lost its inherited resume."
        )
    if (
        parent_job.get("execution_mode") != EXECUTION_MODE
        or resume.get("pointer_closed") is not True
        or resume.get("member_count") != 3
    ):
        raise ScaffoldContractError("Inherited resume contract drifted.")
    archive = resume.get("archive")
    if not isinstance(archive, Mapping):
        raise ScaffoldContractError("Inherited resume lacks an archive.")
    transfer_archive = _repo_binding_from_parent_binding(
        repo_root=repo_root,
        relative_to_parent=archive,
        label=f"{parent_job['execution_id']} inherited resume archive",
        rehash=rehash,
    )
    projected = _canonical_copy(resume)
    projected["archive"] = dict(transfer_archive)
    return projected, transfer_archive


def _effective_source_bindings(
    *,
    repo_root: Path,
    parent_job: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, key in (
        ("source_archive", "effective_source_archive"),
        ("source_manifest", "effective_source_archive_manifest"),
        ("source_delta_receipt", "effective_source_delta_receipt"),
    ):
        raw = parent_job.get(key)
        if not isinstance(raw, Mapping):
            raise ScaffoldContractError(
                f"{parent_job.get('execution_id')} lacks {key}."
            )
        result[role] = _repo_binding_from_parent_binding(
            repo_root=repo_root,
            relative_to_parent=raw,
            label=f"{parent_job['execution_id']} {role}",
            rehash=True,
        )
    return result


def _scientific_anchor(parent_job: Mapping[str, Any]) -> dict[str, Any]:
    protocol = parent_job.get("source_protocol")
    if not isinstance(protocol, Mapping):
        raise ScaffoldContractError("Parent job lacks a source protocol.")
    return {
        "scientific_settings_sha256": str(
            parent_job["scientific_settings_sha256"]
        ),
        "source_protocol_sha256": str(protocol["sha256"]),
        "source_protocol_canonical_sha256": str(
            protocol["canonical_sha256"]
        ),
        "route_contract_sha256": str(
            protocol["route_contract_sha256"]
        ),
    }


def _pending_requirement(
    *, execution_id: str, parent_job: Mapping[str, Any]
) -> dict[str, Any]:
    base = predecessor_requirement(
        execution_id=execution_id, parent_job=parent_job
    )
    unsigned = dict(base)
    unsigned.pop("sha256", None)
    unsigned["scientific_anchor"] = _scientific_anchor(parent_job)
    return digested(unsigned)


def _job_spec(
    *,
    parent_job: Mapping[str, Any],
    parent_job_binding: Mapping[str, Any],
    scientific: Mapping[str, Any],
    resume_input: Mapping[str, Any] | None,
    resume_origin: str,
    requirement: Mapping[str, Any] | None,
    predecessor_binding: Mapping[str, Any] | None,
    effective_sources: Mapping[str, Any],
) -> dict[str, Any]:
    execution_id = str(parent_job["execution_id"])
    ready = resume_input is not None
    predecessor = (
        dict(requirement["predecessor"])
        if requirement is not None
        else None
    )
    return digested(
        {
            "schema": SCAFFOLD_JOB_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "execution_id": execution_id,
            "base_execution_id": parent_job["base_execution_id"],
            "regime_id": parent_job["regime_id"],
            "nph": parent_job["nph"],
            "route_id": parent_job["route_id"],
            "candidate_representation": parent_job[
                "candidate_representation"
            ],
            "insertion_policy": parent_job["insertion_policy"],
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "execution_mode_on_activation": EXECUTION_MODE,
            "status": (
                "ready_authenticated_resume"
                if ready
                else "blocked_missing_authenticated_r50_predecessor"
            ),
            "sealed_parent": {
                "root": SEALED_PARENT_RELATIVE_ROOT,
                "manifest_canonical_sha256": (
                    SEALED_PARENT_MANIFEST_CANONICAL_SHA256
                ),
                "job": _canonical_copy(parent_job_binding),
                "job_schema": parent_job["schema"],
                "job_canonical_sha256": parent_job["sha256"],
            },
            "scientific_settings": _canonical_copy(scientific),
            "scientific_settings_sha256": parent_job[
                "scientific_settings_sha256"
            ],
            "source_protocol": _canonical_copy(
                parent_job["source_protocol"]
            ),
            "source_lock_delta": _canonical_copy(
                parent_job["source_lock_delta"]
            ),
            "scientific_projection_exact": True,
            "only_scientific_change": ONLY_SCIENTIFIC_CHANGE,
            "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
            "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
            "resources": _canonical_copy(parent_job["resources"]),
            "effective_source_family": parent_job[
                "effective_source_family"
            ],
            "effective_sources": _canonical_copy(effective_sources),
            "expected_output_root": parent_job[
                "expected_output_root"
            ],
            "resume_origin": resume_origin,
            "resume_input": (
                _canonical_copy(resume_input)
                if resume_input is not None
                else None
            ),
            "resume_source": (
                _canonical_copy(parent_job.get("resume_source"))
                if resume_origin == "sealed_parent_read_only"
                else (
                    _canonical_copy(
                        predecessor_binding.get("source_receipt")
                    )
                    if predecessor_binding is not None
                    else None
                )
            ),
            "predecessor": predecessor,
            "predecessor_requirement_sha256": (
                requirement["sha256"]
                if requirement is not None
                else None
            ),
            "predecessor_binding_sha256": (
                predecessor_binding["sha256"]
                if predecessor_binding is not None
                else None
            ),
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
        }
    )


def _binding_for_generated(path: Path, *, package_dir: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ScaffoldContractError(f"Generated file is missing: {path}")
    return {
        "path": path.relative_to(package_dir).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def materialize(
    *,
    package_dir: Path = PACKAGE_DIR,
    repo_root: Path | None = None,
    rehash_existing_resumes: bool = False,
) -> dict[str, Any]:
    resolved_repo = (
        repo_root.resolve()
        if repo_root is not None
        else repo_root_from_script(__file__)
    )
    expected_package = package_root(resolved_repo)
    if package_dir.resolve() != expected_package.resolve():
        raise ScaffoldContractError(
            "The canonical scaffold may only materialize at its versioned root."
        )
    if any((package_dir / name).exists() for name in GENERATED_FILES):
        raise ScaffoldContractError(
            "Scaffold is already materialized; refusing in-place rewrite."
        )
    if (package_dir / JOBS_DIR).exists() or (
        package_dir / PREDECESSOR_PLACEHOLDERS_DIR
    ).exists():
        raise ScaffoldContractError(
            "Generated job/placeholder directories already exist."
        )
    for forbidden in ("submit.sub", "authority", "authorization_manifest.json"):
        if (package_dir / forbidden).exists():
            raise ScaffoldContractError(
                "Inert scaffold unexpectedly contains activation state."
            )

    parent_manifest, parent_jobs = load_sealed_parent_jobs(
        resolved_repo, rehash_jobs=True
    )
    parent_bindings = _parent_job_bindings(parent_manifest)
    jobs_dir = package_dir / JOBS_DIR
    placeholders_dir = package_dir / PREDECESSOR_PLACEHOLDERS_DIR
    jobs_dir.mkdir(parents=True, exist_ok=False)
    placeholders_dir.mkdir(parents=True, exist_ok=False)

    job_records: list[dict[str, Any]] = []
    requirements: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    inherited_count = 0
    newly_bound_count = 0
    missing_count = 0

    for proc_id, parent_job in enumerate(parent_jobs):
        execution_id = str(parent_job["execution_id"])
        scientific = validate_scientific_projection(parent_job)
        effective_sources = _effective_source_bindings(
            repo_root=resolved_repo, parent_job=parent_job
        )
        requirement: dict[str, Any] | None = None
        predecessor_binding: dict[str, Any] | None = None
        if execution_id in PENDING_PREDECESSORS:
            requirement = _pending_requirement(
                execution_id=execution_id, parent_job=parent_job
            )
            requirements.append(requirement)
            _write_json(
                placeholders_dir / f"{execution_id}.json",
                requirement,
            )
            predecessor_binding = None
            resume_input = None
            transfer_resume = {
                "path": requirement["resume_archive_path"],
                "sha256": None,
                "size_bytes": None,
                "status": "missing_fail_closed",
            }
            resume_origin = "pending_external_predecessor_binding"
            missing_count += 1
        else:
            resume_input, transfer_resume = _resume_input_from_parent(
                repo_root=resolved_repo,
                parent_job=parent_job,
                rehash=rehash_existing_resumes,
            )
            transfer_resume["status"] = "ready"
            resume_origin = "sealed_parent_read_only"
            inherited_count += 1

        spec = _job_spec(
            parent_job=parent_job,
            parent_job_binding=parent_bindings[execution_id],
            scientific=scientific,
            resume_input=resume_input,
            resume_origin=resume_origin,
            requirement=requirement,
            predecessor_binding=predecessor_binding,
            effective_sources=effective_sources,
        )
        job_path = jobs_dir / f"{execution_id}.json"
        _write_json(job_path, spec)
        job_binding = _binding_for_generated(
            job_path, package_dir=package_dir
        )
        job_binding["canonical_sha256"] = spec["sha256"]
        job_binding["execution_id"] = execution_id
        job_binding["status"] = spec["status"]
        job_records.append(job_binding)

        transfer_inputs = [
            {
                "role": "runtime_bundle",
                "path": f"{PACKAGE_RELATIVE_ROOT}/{RUNTIME_BUNDLE_RELATIVE}",
                "status": "missing_until_runtime_activation",
            },
            {
                "role": "job_spec",
                "path": f"{PACKAGE_RELATIVE_ROOT}/{job_binding['path']}",
                "sha256": job_binding["sha256"],
                "size_bytes": job_binding["size_bytes"],
                "status": "ready",
            },
            dict(effective_sources["source_archive"], role="source_archive", status="ready"),
            dict(effective_sources["source_manifest"], role="source_manifest", status="ready"),
            dict(
                effective_sources["source_delta_receipt"],
                role="source_delta_receipt",
                status="ready",
            ),
            dict(transfer_resume, role="resume_archive"),
            {
                "role": "execution_authorization",
                "path": (
                    f"{PACKAGE_RELATIVE_ROOT}/authorizations/"
                    f"{execution_id}.json"
                ),
                "status": "missing_until_activation",
            },
        ]
        transfer_rows.append(
            {
                "proc_id": proc_id,
                "execution_id": execution_id,
                "status": (
                    "blocked_missing_resume"
                    if resume_input is None
                    else "blocked_global_activation_gates"
                ),
                "resume_origin": resume_origin,
                "transfer_inputs": transfer_inputs,
                "resume_archive_count": 1,
                "aggregate_resume_directory_transferred": False,
                "request_cpus": parent_job["resources"]["request_cpus"],
                "request_memory_mb": parent_job["resources"][
                    "request_memory_mb"
                ],
                "request_disk_mb": parent_job["resources"][
                    "request_disk_mb"
                ],
                "max_runtime_seconds": parent_job["resources"][
                    "max_runtime_seconds"
                ],
            }
        )

    if (
        inherited_count != INHERITED_RESUME_COUNT
        or newly_bound_count + missing_count != PENDING_RESUME_COUNT
        or len(job_records) != CELL_COUNT
    ):
        raise ScaffoldContractError("Resume inventory cardinality drifted.")

    requirements_payload = digested(
        {
            "schema": PREDECESSOR_REQUIREMENT_SCHEMA + "_inventory",
            "package_id": PACKAGE_ID,
            "required_count": PENDING_RESUME_COUNT,
            "missing_count": missing_count,
            "bound_count": newly_bound_count,
            "requirements": requirements,
            "status": (
                "blocked_missing_predecessors"
                if missing_count
                else "passed_all_predecessors_bound"
            ),
        }
    )
    _write_json(
        package_dir / PREDECESSOR_REQUIREMENTS_NAME,
        requirements_payload,
    )

    plan = digested(
        {
            "schema": PACKAGE_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "cell_count": CELL_COUNT,
            "target_authenticated_resume_count": TARGET_RESUME_COUNT,
            "ready_authenticated_resume_count": (
                inherited_count + newly_bound_count
            ),
            "inherited_read_only_resume_count": inherited_count,
            "newly_bound_resume_count": newly_bound_count,
            "missing_resume_count": missing_count,
            "execution_mode_for_all_rows": EXECUTION_MODE,
            "only_scientific_change": ONLY_SCIENTIFIC_CHANGE,
            "scientific_settings_exact_parent_count": CELL_COUNT,
            "sealed_parent_binding_mode": "exact_bytes_not_rederived",
            "known_parent_dynamic_rederivation_drift": {
                "command": "validate_operational_overlay_v2.py --metadata-only",
                "status": "known_failed_out_of_scope",
                "observed_error": (
                    "core__weak_weak__nph3__ra_macro_always__r70 "
                    "job row field drifted: resume_source"
                ),
                "sealed_parent_rewritten": False,
                "semantic_rederivation_used": False,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "submit_descriptor_present": False,
            "submission_blockers": [
                *(
                    ["nine_authenticated_r50_predecessor_bindings_missing"]
                    if missing_count
                    else []
                ),
                "r70_resource_envelopes_not_demonstrated",
                "row_runtime_bundle_absent",
                "execution_authorizations_absent",
                "remote_image_proof_absent",
                "submit_descriptor_intentionally_absent",
            ],
            "jobs": job_records,
            "status": (
                "passed_inert_scaffold_missing_predecessors"
                if missing_count
                else "passed_inert_scaffold_awaiting_activation"
            ),
        }
    )
    _write_json(package_dir / PACKAGE_PLAN_NAME, plan)

    transfer_plan = digested(
        {
            "schema": TRANSFER_PLAN_SCHEMA,
            "package_id": PACKAGE_ID,
            "cell_count": CELL_COUNT,
            "row_count": len(transfer_rows),
            "transfer_shape": "one_exact_resume_archive_per_row",
            "aggregate_resume_directory_transferred": False,
            "aggregate_parent_package_transferred": False,
            "rows": transfer_rows,
            "status": "passed_row_sharded_inert_plan",
        }
    )
    _write_json(package_dir / TRANSFER_PLAN_NAME, transfer_plan)
    queue_lines = []
    for row in transfer_rows:
        by_role = {item["role"]: item for item in row["transfer_inputs"]}
        queue_lines.append(
            "\t".join(
                str(value)
                for value in (
                    row["proc_id"],
                    row["execution_id"],
                    row["status"],
                    by_role["job_spec"]["path"],
                    by_role["resume_archive"]["path"],
                    by_role["source_archive"]["path"],
                    row["request_cpus"],
                    row["request_memory_mb"],
                    row["request_disk_mb"],
                    row["max_runtime_seconds"],
                )
            )
        )
    _exclusive_write(
        package_dir / TRANSFER_QUEUE_NAME,
        ("\n".join(queue_lines) + "\n").encode("utf-8"),
    )

    controls = []
    for name in CONTROL_FILES:
        path = package_dir / name
        controls.append(_binding_for_generated(path, package_dir=package_dir))
    generated = {
        name: _binding_for_generated(package_dir / name, package_dir=package_dir)
        for name in GENERATED_FILES
        if name != SCAFFOLD_MANIFEST_NAME
    }
    placeholder_bindings = [
        _binding_for_generated(path, package_dir=package_dir)
        for path in sorted(placeholders_dir.glob("*.json"))
    ]
    manifest = digested(
        {
            "schema": SCAFFOLD_MANIFEST_SCHEMA,
            "package_id": PACKAGE_ID,
            "campaign_id": CAMPAIGN_ID,
            "run_class": RUN_CLASS,
            "execution_target": EXECUTION_TARGET,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "cell_count": CELL_COUNT,
            "target_authenticated_resume_count": TARGET_RESUME_COUNT,
            "ready_authenticated_resume_count": (
                inherited_count + newly_bound_count
            ),
            "missing_resume_count": missing_count,
            "sealed_parent": {
                "root": SEALED_PARENT_RELATIVE_ROOT,
                "manifest_path": SEALED_PARENT_MANIFEST_NAME,
                "manifest_file_sha256": (
                    SEALED_PARENT_MANIFEST_FILE_SHA256
                ),
                "manifest_canonical_sha256": (
                    SEALED_PARENT_MANIFEST_CANONICAL_SHA256
                ),
                "binding_mode": "exact_sealed_bytes_only",
                "dynamic_rederivation_required": False,
            },
            "scientific_settings_exact_parent_count": CELL_COUNT,
            "jobs": job_records,
            "predecessor_placeholders": placeholder_bindings,
            "control_files": controls,
            "external_control_dependencies": [
                dict(CONTROLLED_CYCLE_VALIDATOR_BINDING)
            ],
            "generated": generated,
            "row_transfer_plan": generated[TRANSFER_PLAN_NAME],
            "aggregate_resume_directory_transferred": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_ready": False,
            "submitted": False,
            "submit_descriptor_present": False,
            "status": (
                "passed_inert_scaffold_missing_9_predecessors"
                if missing_count == PENDING_RESUME_COUNT
                else "passed_inert_scaffold_partial_predecessors"
                if missing_count
                else "passed_inert_scaffold_awaiting_activation"
            ),
        }
    )
    _write_json(package_dir / SCAFFOLD_MANIFEST_NAME, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rehash-existing-resumes",
        action="store_true",
        help="Rehash all 27 large inherited resume archives.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = materialize(
            rehash_existing_resumes=args.rehash_existing_resumes
        )
    except (OSError, ScaffoldContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(manifest).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
