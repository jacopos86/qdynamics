#!/usr/bin/env python3
"""Validate the inert, row-sharded RA r50->r70 continuation scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from scaffold_contract import (  # noqa: E402
    CAMPAIGN_ID,
    CELL_COUNT,
    CONTROL_FILES,
    CONTROLLED_CYCLE_VALIDATOR_BINDING,
    EXECUTION_MODE,
    GENERATED_FILES,
    INHERITED_RESUME_COUNT,
    JOBS_DIR,
    NEW_RESUME_INPUTS_DIR,
    ONLY_SCIENTIFIC_CHANGE,
    PACKAGE_ID,
    PACKAGE_PLAN_NAME,
    PACKAGE_PLAN_SCHEMA,
    PACKAGE_RELATIVE_ROOT,
    PENDING_PREDECESSORS,
    PENDING_RESUME_COUNT,
    PREDECESSOR_PLACEHOLDERS_DIR,
    PREDECESSOR_REQUIREMENT_SCHEMA,
    PREDECESSOR_REQUIREMENTS_NAME,
    RUN_CLASS,
    SCAFFOLD_JOB_SCHEMA,
    SCAFFOLD_MANIFEST_NAME,
    SCAFFOLD_MANIFEST_SCHEMA,
    SEALED_PARENT_MANIFEST_CANONICAL_SHA256,
    SEALED_PARENT_MANIFEST_FILE_SHA256,
    SEALED_PARENT_RELATIVE_ROOT,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    TARGET_RESUME_COUNT,
    TRANSFER_PLAN_NAME,
    TRANSFER_PLAN_SCHEMA,
    TRANSFER_QUEUE_NAME,
    ScaffoldContractError,
    canonical_json_bytes,
    canonical_sha256,
    load_json,
    load_sealed_parent_jobs,
    parent_root,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    transfer_path_is_regular_file,
    validate_scientific_projection,
    verify_exact_binding,
    verify_self_digest,
    verify_controlled_cycle_dependency,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScaffoldContractError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScaffoldContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScaffoldContractError(f"{label} must be a list.")
    return value


def _verify_package_binding(
    binding: Mapping[str, Any], *, label: str
) -> Path:
    relative = safe_relative_path(binding.get("path"), label=f"{label} path")
    path = PACKAGE_DIR / relative
    verify_exact_binding(path, binding, label=label, rehash=True)
    return path


def _parent_bindings(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["execution_id"]): row
        for row in _sequence(manifest.get("jobs"), label="parent jobs")
        if isinstance(row, Mapping)
    }


def _normalized_parent_resume(
    *, repo_root: Path, parent_job: Mapping[str, Any]
) -> dict[str, Any]:
    resume = _mapping(parent_job.get("resume_input"), label="parent resume")
    result = json.loads(canonical_json_bytes(resume))
    archive = _mapping(result.get("archive"), label="parent resume archive")
    source_path = parent_root(repo_root) / safe_relative_path(
        archive.get("path"), label="parent resume archive path"
    )
    archive["path"] = source_path.relative_to(repo_root).as_posix()
    return result


def _validate_transfer_plan(
    *,
    transfer: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
) -> None:
    verify_self_digest(transfer, label="row transfer plan")
    rows = _sequence(transfer.get("rows"), label="transfer rows")
    _require(
        transfer.get("schema") == TRANSFER_PLAN_SCHEMA
        and transfer.get("package_id") == PACKAGE_ID
        and transfer.get("cell_count") == CELL_COUNT
        and transfer.get("row_count") == CELL_COUNT
        and transfer.get("transfer_shape")
        == "one_exact_resume_archive_per_row"
        and transfer.get("aggregate_resume_directory_transferred") is False
        and transfer.get("aggregate_parent_package_transferred") is False
        and len(rows) == CELL_COUNT,
        "Row transfer plan header drifted.",
    )
    seen: set[str] = set()
    queue_expected: list[str] = []
    for expected_proc, raw in enumerate(rows):
        row = _mapping(raw, label=f"transfer row {expected_proc}")
        execution_id = str(row.get("execution_id", ""))
        inputs = _sequence(
            row.get("transfer_inputs"), label=f"{execution_id} transfer inputs"
        )
        by_role = {
            str(item["role"]): item
            for item in inputs
            if isinstance(item, Mapping)
        }
        _require(
            execution_id in jobs
            and execution_id not in seen
            and row.get("proc_id") == expected_proc
            and row.get("resume_archive_count") == 1
            and row.get("aggregate_resume_directory_transferred") is False
            and set(by_role)
            == {
                "runtime_bundle",
                "job_spec",
                "source_archive",
                "source_manifest",
                "source_delta_receipt",
                "resume_archive",
                "execution_authorization",
            },
            f"{execution_id} transfer inventory drifted.",
        )
        seen.add(execution_id)
        for role, item in by_role.items():
            path = str(item.get("path", ""))
            safe_relative_path(path, label=f"{execution_id} {role} transfer path")
            _require(
                transfer_path_is_regular_file(path),
                f"{execution_id} {role} is not an exact file transfer.",
            )
            _require(
                path
                not in {
                    f"{SEALED_PARENT_RELATIVE_ROOT}/resume_inputs",
                    f"{PACKAGE_RELATIVE_ROOT}/{NEW_RESUME_INPUTS_DIR}",
                    SEALED_PARENT_RELATIVE_ROOT,
                    PACKAGE_RELATIVE_ROOT,
                },
                f"{execution_id} attempts an aggregate directory transfer.",
            )
        resume = by_role["resume_archive"]
        _require(
            str(resume["path"]).endswith(f"/{execution_id}.tar.gz"),
            f"{execution_id} does not own exactly one row resume archive.",
        )
        job = jobs[execution_id]
        expected_status = (
            "missing_fail_closed"
            if job.get("resume_input") is None
            else "ready"
        )
        _require(
            resume.get("status") == expected_status,
            f"{execution_id} resume transfer readiness drifted.",
        )
        queue_expected.append(
            "\t".join(
                str(value)
                for value in (
                    expected_proc,
                    execution_id,
                    row["status"],
                    by_role["job_spec"]["path"],
                    resume["path"],
                    by_role["source_archive"]["path"],
                    row["request_cpus"],
                    row["request_memory_mb"],
                    row["request_disk_mb"],
                    row["max_runtime_seconds"],
                )
            )
        )
    _require(seen == set(jobs), "Transfer plan job coverage drifted.")
    observed_queue = (PACKAGE_DIR / TRANSFER_QUEUE_NAME).read_text(
        encoding="utf-8"
    ).splitlines()
    _require(
        observed_queue == queue_expected,
        "Row transfer TSV drifted from the JSON plan.",
    )


def validate_scaffold(
    *, rehash_existing_resumes: bool = False
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    for forbidden in (
        "submit.sub",
        "authority",
        "authorization_manifest.json",
        "submission_receipt.json",
    ):
        _require(
            not (PACKAGE_DIR / forbidden).exists(),
            "Inert scaffold gained live activation/submission state.",
        )

    manifest = load_json(
        PACKAGE_DIR / SCAFFOLD_MANIFEST_NAME,
        label="scaffold manifest",
    )
    manifest_digest = verify_self_digest(
        manifest, label="scaffold manifest"
    )
    _require(
        manifest.get("schema") == SCAFFOLD_MANIFEST_SCHEMA
        and manifest.get("package_id") == PACKAGE_ID
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("run_class") == RUN_CLASS
        and manifest.get("cell_count") == CELL_COUNT
        and manifest.get("target_authenticated_resume_count")
        == TARGET_RESUME_COUNT
        and manifest.get("ready_authenticated_resume_count")
        == INHERITED_RESUME_COUNT
        and manifest.get("missing_resume_count") == PENDING_RESUME_COUNT
        and manifest.get("scientific_settings_exact_parent_count")
        == CELL_COUNT
        and manifest.get("aggregate_resume_directory_transferred") is False
        and manifest.get("execution_authorized") is False
        and manifest.get("submission_authorized") is False
        and manifest.get("submission_ready") is False
        and manifest.get("submitted") is False
        and manifest.get("submit_descriptor_present") is False
        and manifest.get("status")
        == "passed_inert_scaffold_missing_9_predecessors",
        "Scaffold manifest is not the expected inert 27+9 state.",
    )
    parent_binding = _mapping(
        manifest.get("sealed_parent"), label="sealed parent binding"
    )
    _require(
        parent_binding.get("root") == SEALED_PARENT_RELATIVE_ROOT
        and parent_binding.get("manifest_file_sha256")
        == SEALED_PARENT_MANIFEST_FILE_SHA256
        and parent_binding.get("manifest_canonical_sha256")
        == SEALED_PARENT_MANIFEST_CANONICAL_SHA256
        and parent_binding.get("binding_mode")
        == "exact_sealed_bytes_only"
        and parent_binding.get("dynamic_rederivation_required") is False,
        "Sealed-parent byte binding drifted.",
    )

    for binding in _sequence(
        manifest.get("control_files"), label="control files"
    ):
        path = _verify_package_binding(
            _mapping(binding, label="control binding"),
            label="control file",
        )
        _require(path.name in CONTROL_FILES, "Unknown control file bound.")
    external_dependencies = _sequence(
        manifest.get("external_control_dependencies"),
        label="external control dependencies",
    )
    _require(
        external_dependencies == [CONTROLLED_CYCLE_VALIDATOR_BINDING]
        and verify_controlled_cycle_dependency()
        == CONTROLLED_CYCLE_VALIDATOR_BINDING,
        "External predecessor-authentication dependency drifted.",
    )
    generated = _mapping(
        manifest.get("generated"), label="generated bindings"
    )
    _require(
        set(generated) == set(GENERATED_FILES).difference({SCAFFOLD_MANIFEST_NAME}),
        "Generated document inventory drifted.",
    )
    for name, binding in generated.items():
        path = _verify_package_binding(
            _mapping(binding, label=f"{name} binding"), label=name
        )
        _require(path.name == name, f"{name} binding path drifted.")

    parent_manifest, parent_jobs_list = load_sealed_parent_jobs(
        repo_root, rehash_jobs=True
    )
    parent_jobs = {
        str(job["execution_id"]): job for job in parent_jobs_list
    }
    parent_job_bindings = _parent_bindings(parent_manifest)
    job_bindings = _sequence(manifest.get("jobs"), label="scaffold jobs")
    _require(len(job_bindings) == CELL_COUNT, "Scaffold job count drifted.")
    jobs: dict[str, Mapping[str, Any]] = {}
    inherited_ready = 0
    missing = 0
    for raw_binding in job_bindings:
        binding = _mapping(raw_binding, label="scaffold job binding")
        path = _verify_package_binding(binding, label="scaffold job")
        job = load_json(path, label="scaffold job")
        canonical = verify_self_digest(job, label="scaffold job")
        execution_id = str(job.get("execution_id", ""))
        parent_job = _mapping(
            parent_jobs.get(execution_id), label="sealed parent job"
        )
        scientific = validate_scientific_projection(parent_job)
        _require(
            binding.get("execution_id") == execution_id
            and binding.get("canonical_sha256") == canonical
            and job.get("schema") == SCAFFOLD_JOB_SCHEMA
            and job.get("package_id") == PACKAGE_ID
            and job.get("campaign_id") == CAMPAIGN_ID
            and job.get("run_class") == RUN_CLASS
            and job.get("source_horizon") == SOURCE_HORIZON
            and job.get("target_horizon") == TARGET_HORIZON
            and job.get("execution_mode_on_activation") == EXECUTION_MODE
            and job.get("only_scientific_change") == ONLY_SCIENTIFIC_CHANGE
            and job.get("scientific_projection_exact") is True
            and job.get("scientific_settings") == scientific
            and job.get("scientific_settings_sha256")
            == canonical_sha256(scientific)
            and job.get("source_protocol") == parent_job.get("source_protocol")
            and job.get("source_lock_delta") == parent_job.get("source_lock_delta")
            and job.get("resources") == parent_job.get("resources")
            and job.get("execution_authorized") is False
            and job.get("submission_authorized") is False
            and job.get("submission_ready") is False
            and job.get("submitted") is False,
            f"{execution_id} scientific or inert job contract drifted.",
        )
        sealed = _mapping(job.get("sealed_parent"), label="job sealed parent")
        _require(
            sealed.get("manifest_canonical_sha256")
            == SEALED_PARENT_MANIFEST_CANONICAL_SHA256
            and sealed.get("job") == parent_job_bindings[execution_id]
            and sealed.get("job_canonical_sha256") == parent_job["sha256"],
            f"{execution_id} sealed job binding drifted.",
        )
        if execution_id in PENDING_PREDECESSORS:
            requirement_path = (
                PACKAGE_DIR
                / PREDECESSOR_PLACEHOLDERS_DIR
                / f"{execution_id}.json"
            )
            requirement = load_json(
                requirement_path,
                label=f"{execution_id} predecessor placeholder",
            )
            verify_self_digest(requirement, label="predecessor placeholder")
            _require(
                requirement.get("schema") == PREDECESSOR_REQUIREMENT_SCHEMA
                and requirement.get("execution_id") == execution_id
                and requirement.get("predecessor")
                == PENDING_PREDECESSORS[execution_id]
                and requirement.get("status") == "missing_fail_closed"
                and job.get("resume_input") is None
                and job.get("resume_origin")
                == "pending_external_predecessor_binding"
                and job.get("status")
                == "blocked_missing_authenticated_r50_predecessor"
                and job.get("predecessor_requirement_sha256")
                == requirement["sha256"]
                and job.get("predecessor_binding_sha256") is None,
                f"{execution_id} missing predecessor gate drifted.",
            )
            missing += 1
        else:
            observed_resume = _mapping(
                job.get("resume_input"), label="inherited resume"
            )
            expected_resume = _normalized_parent_resume(
                repo_root=repo_root, parent_job=parent_job
            )
            archive = _mapping(
                observed_resume.get("archive"), label="inherited archive"
            )
            archive_path = repo_root / safe_relative_path(
                archive.get("path"), label="inherited archive path"
            )
            verify_exact_binding(
                archive_path,
                archive,
                label=f"{execution_id} inherited resume archive",
                rehash=rehash_existing_resumes,
            )
            _require(
                observed_resume == expected_resume
                and job.get("resume_origin") == "sealed_parent_read_only"
                and job.get("status") == "ready_authenticated_resume"
                and job.get("predecessor") is None,
                f"{execution_id} inherited read-only resume drifted.",
            )
            inherited_ready += 1
        jobs[execution_id] = job
    _require(
        inherited_ready == INHERITED_RESUME_COUNT
        and missing == PENDING_RESUME_COUNT
        and set(jobs) == set(parent_jobs),
        "Scaffold 27+9 partition drifted.",
    )

    requirements = load_json(
        PACKAGE_DIR / PREDECESSOR_REQUIREMENTS_NAME,
        label="predecessor requirement inventory",
    )
    verify_self_digest(requirements, label="predecessor requirement inventory")
    _require(
        requirements.get("required_count") == PENDING_RESUME_COUNT
        and requirements.get("missing_count") == PENDING_RESUME_COUNT
        and requirements.get("bound_count") == 0
        and requirements.get("status") == "blocked_missing_predecessors"
        and len(_sequence(requirements.get("requirements"), label="requirements"))
        == PENDING_RESUME_COUNT,
        "Predecessor requirement inventory drifted.",
    )

    plan = load_json(PACKAGE_DIR / PACKAGE_PLAN_NAME, label="package plan")
    verify_self_digest(plan, label="package plan")
    drift = _mapping(
        plan.get("known_parent_dynamic_rederivation_drift"),
        label="known parent dynamic rederivation drift",
    )
    _require(
        plan.get("schema") == PACKAGE_PLAN_SCHEMA
        and plan.get("package_id") == PACKAGE_ID
        and plan.get("cell_count") == CELL_COUNT
        and plan.get("target_authenticated_resume_count") == TARGET_RESUME_COUNT
        and plan.get("ready_authenticated_resume_count")
        == INHERITED_RESUME_COUNT
        and plan.get("missing_resume_count") == PENDING_RESUME_COUNT
        and plan.get("execution_mode_for_all_rows") == EXECUTION_MODE
        and plan.get("sealed_parent_binding_mode")
        == "exact_bytes_not_rederived"
        and drift.get("status") == "known_failed_out_of_scope"
        and drift.get("semantic_rederivation_used") is False
        and drift.get("sealed_parent_rewritten") is False
        and "resume_source" in str(drift.get("observed_error"))
        and plan.get("submission_ready") is False
        and plan.get("submit_descriptor_present") is False,
        "Package plan or parent-validator drift record changed.",
    )
    transfer = load_json(
        PACKAGE_DIR / TRANSFER_PLAN_NAME, label="row transfer plan"
    )
    _validate_transfer_plan(transfer=transfer, jobs=jobs)

    return {
        "status": "passed_inert_scaffold_missing_9_predecessors",
        "package_id": PACKAGE_ID,
        "manifest_sha256": manifest_digest,
        "sealed_parent_manifest_file_sha256": (
            SEALED_PARENT_MANIFEST_FILE_SHA256
        ),
        "sealed_parent_manifest_canonical_sha256": (
            SEALED_PARENT_MANIFEST_CANONICAL_SHA256
        ),
        "scientific_settings_exact_parent_count": CELL_COUNT,
        "ready_authenticated_resume_count": inherited_ready,
        "missing_authenticated_resume_count": missing,
        "row_sharded_transfer_count": CELL_COUNT,
        "aggregate_resume_directory_transferred": False,
        "submission_ready": False,
    }


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
        result = validate_scaffold(
            rehash_existing_resumes=args.rehash_existing_resumes
        )
    except (OSError, ScaffoldContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
