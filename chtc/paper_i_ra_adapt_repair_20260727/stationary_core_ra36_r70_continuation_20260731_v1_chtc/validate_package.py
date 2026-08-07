#!/usr/bin/env python3
"""Validate the inert, collision-blocked stationary-core RA r70 package."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    ACTIVE_GRADIENT_POLICY,
    AUTHORIZATION_SCHEMA,
    CAMPAIGN_ID,
    CELL_COUNT,
    COLLISION_CLUSTER_ID,
    COLLISION_PROC_IDS,
    COLLISION_STATUS_NAME,
    COLLISION_STATUS_SCHEMA,
    CONTROL_FILES,
    EXECUTION_PLAN_NAME,
    EXECUTION_PLAN_SCHEMA,
    FRESH_COUNT,
    JOB_SCHEMA,
    PACKAGE_ID,
    PACKAGE_MANIFEST_NAME,
    PACKAGE_MANIFEST_SCHEMA,
    QUEUE_NAME,
    RESOURCE_WEIGHTING_SCOPE,
    RESUME_COUNT,
    RESUME_INPUTS_NAME,
    RESUME_INPUTS_SCHEMA,
    SOURCE_ARCHIVES_NAME,
    SOURCE_ARCHIVES_SCHEMA,
    SOURCE_FAMILIES,
    SOURCE_HORIZON,
    SOURCE_LOCK_AUDIT_NAME,
    SOURCE_REPORT_RELATIVE,
    TARGET_HORIZON,
    PackageContractError,
    collision_map,
    load_json,
    planned_rows,
    repo_root_from_script,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PackageContractError(f"{label} must be a list.")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageContractError(message)


def _package_path(value: Any, *, label: str) -> Path:
    relative = safe_relative_path(value, label=label)
    path = PACKAGE_DIR / relative
    try:
        path.resolve().relative_to(PACKAGE_DIR.resolve())
    except ValueError as exc:
        raise PackageContractError(
            f"{label} escaped the package."
        ) from exc
    return path


def _verify_file_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    hash_payload: bool = True,
) -> Path:
    path = _package_path(binding.get("path"), label=f"{label} path")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != int(binding.get("size_bytes", -1))
    ):
        raise PackageContractError(f"{label} size or type drifted.")
    if hash_payload and sha256_file(path) != binding.get("sha256"):
        raise PackageContractError(f"{label} digest drifted.")
    return path


def _scan_tar(
    archive_path: Path,
    *,
    expected_members: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            expected = expected_members.get(member.name)
            if (
                expected is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size
                != int(expected.get("size_bytes", -1))
            ):
                raise PackageContractError(
                    f"{label} contains an unexpected or unsafe member: "
                    f"{member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"{label} member is unreadable: {member.name}"
                )
            import hashlib

            digest = hashlib.sha256()
            size = 0
            for block in iter(
                lambda: stream.read(1024 * 1024), b""
            ):
                digest.update(block)
                size += len(block)
            if (
                size != member.size
                or digest.hexdigest() != expected.get("sha256")
            ):
                raise PackageContractError(
                    f"{label} member digest drifted: {member.name}"
                )
            observed.add(member.name)
    if observed != set(expected_members):
        missing = sorted(set(expected_members).difference(observed))
        raise PackageContractError(
            f"{label} is missing members: {missing}"
        )


def _validate_source_archives(
    *,
    payload: Mapping[str, Any],
    repo_root: Path,
    full_archive_scan: bool,
) -> None:
    verify_self_digest(payload, label="source archives")
    families = _mapping(
        payload.get("families"), label="source archive families"
    )
    _require(
        payload.get("schema") == SOURCE_ARCHIVES_SCHEMA
        and payload.get("package_id") == PACKAGE_ID
        and payload.get("status") == "passed"
        and int(payload.get("family_count", -1))
        == len(SOURCE_FAMILIES)
        and set(families) == set(SOURCE_FAMILIES),
        "Source-archive index identity drifted.",
    )
    for family_id, expected_family in SOURCE_FAMILIES.items():
        row = _mapping(
            families[family_id], label=f"{family_id} source row"
        )
        archive = _verify_file_binding(
            _mapping(
                row.get("packaged_archive"),
                label=f"{family_id} packaged archive",
            ),
            label=f"{family_id} packaged archive",
        )
        manifest_path = _verify_file_binding(
            _mapping(
                row.get("packaged_manifest"),
                label=f"{family_id} packaged manifest",
            ),
            label=f"{family_id} packaged manifest",
        )
        source_root = (
            repo_root / str(expected_family["source_package_root"])
        )
        original_archive = source_root / "source_locked.tar.gz"
        original_manifest = source_root / "source_archive_manifest.json"
        _require(
            row.get("source_package_root")
            == expected_family["source_package_root"]
            and row.get("exact_copy") is True
            and original_archive.is_file()
            and original_manifest.is_file()
            and sha256_file(original_archive)
            == row["original_archive"]["sha256"]
            == row["packaged_archive"]["sha256"]
            and sha256_file(original_manifest)
            == row["original_manifest"]["sha256"]
            == row["packaged_manifest"]["sha256"],
            f"{family_id} is not an exact source-package copy.",
        )
        source_manifest = load_json(
            manifest_path, label=f"{family_id} source manifest"
        )
        verify_self_digest(
            source_manifest, label=f"{family_id} source manifest"
        )
        manifest_archive = _mapping(
            source_manifest.get("archive"),
            label=f"{family_id} manifest archive",
        )
        members = _sequence(
            source_manifest.get("members"),
            label=f"{family_id} source members",
        )
        member_map = {
            safe_relative_path(
                item.get("path"),
                label=f"{family_id} source member path",
            ).as_posix(): _mapping(
                item, label=f"{family_id} source member"
            )
            for item in members
            if isinstance(item, Mapping)
        }
        _require(
            manifest_archive.get("sha256")
            == row["packaged_archive"]["sha256"]
            and int(manifest_archive.get("size_bytes", -1))
            == archive.stat().st_size
            and len(member_map)
            == len(members)
            == int(source_manifest.get("member_count", -1))
            == int(row.get("member_count", -1)),
            f"{family_id} source manifest closure drifted.",
        )
        if full_archive_scan:
            _scan_tar(
                archive,
                expected_members=member_map,
                label=f"{family_id} source archive",
            )


def _validate_resume_inputs(
    *,
    payload: Mapping[str, Any],
    resume_ids: set[str],
    full_archive_scan: bool,
) -> None:
    verify_self_digest(payload, label="resume inputs")
    cells = _mapping(payload.get("cells"), label="resume cells")
    _require(
        payload.get("schema") == RESUME_INPUTS_SCHEMA
        and payload.get("package_id") == PACKAGE_ID
        and payload.get("status") == "passed"
        and int(payload.get("resume_cell_count", -1))
        == RESUME_COUNT
        and int(payload.get("source_horizon", -1))
        == SOURCE_HORIZON
        and int(payload.get("target_horizon", -1))
        == TARGET_HORIZON
        and set(cells) == resume_ids,
        "Resume-input index identity drifted.",
    )
    for execution_id in sorted(cells):
        cell = _mapping(
            cells[execution_id], label=f"{execution_id} resume cell"
        )
        archive_binding = _mapping(
            cell.get("archive"),
            label=f"{execution_id} resume archive",
        )
        archive = _verify_file_binding(
            archive_binding,
            label=f"{execution_id} resume archive",
            hash_payload=full_archive_scan,
        )
        members = _sequence(
            cell.get("members"),
            label=f"{execution_id} resume members",
        )
        member_map = {
            safe_relative_path(
                item.get("path"),
                label=f"{execution_id} resume member path",
            ).as_posix(): _mapping(
                item, label=f"{execution_id} resume member"
            )
            for item in members
            if isinstance(item, Mapping)
        }
        roles = {str(item.get("role")) for item in members}
        authentication = _mapping(
            cell.get("authentication"),
            label=f"{execution_id} authentication",
        )
        checkpoint = member_map.get(str(cell.get("checkpoint_path")))
        _require(
            int(cell.get("member_count", -1)) == 3
            and len(member_map) == len(members) == 3
            and roles
            == {
                "checkpoint",
                "estimator_ledger_checkpoint",
                "verified_resume_sidecar",
            }
            and cell.get("pointer_closed") is True
            and cell.get("superseded_sidecars_retained") is False
            and isinstance(checkpoint, Mapping)
            and checkpoint.get("role") == "checkpoint"
            and checkpoint.get("sha256")
            == cell.get("checkpoint_sha256")
            == cell.get("source_checkpoint_sha256")
            and int(authentication.get("checkpoint_depth", -1))
            == SOURCE_HORIZON
            and int(authentication.get("history_count", -1))
            == SOURCE_HORIZON
            and int(
                authentication.get(
                    "active_prefix_checkpoint_count", -1
                )
            )
            == SOURCE_HORIZON
            and authentication.get("history_checkpoint_complete")
            is True
            and authentication.get("strict_replay_passed") is True,
            f"{execution_id} compact resume closure drifted.",
        )
        if full_archive_scan:
            _scan_tar(
                archive,
                expected_members=member_map,
                label=f"{execution_id} resume archive",
            )


def _validate_collision(
    *,
    payload: Mapping[str, Any],
    repo_root: Path,
    fresh_ids: set[str],
) -> None:
    verify_self_digest(payload, label="collision status")
    rows = _sequence(payload.get("rows"), label="collision rows")
    expected = collision_map(repo_root)
    observed = {
        str(row["base_execution_id"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    _require(
        payload.get("schema") == COLLISION_STATUS_SCHEMA
        and payload.get("package_id") == PACKAGE_ID
        and payload.get("status") == "blocked"
        and payload.get("blocking") is True
        and int(payload.get("cluster_id", -1))
        == COLLISION_CLUSTER_ID
        and payload.get("proc_ids") == list(COLLISION_PROC_IDS)
        and set(observed) == fresh_ids == set(expected)
        and payload.get("external_state_revalidation_required")
        is True
        and payload.get("submit_descriptor_present") is False
        and payload.get("may_submit") is False
        and payload.get("may_supersede_predecessors") is False
        and payload.get("may_remove_predecessors") is False,
        "Collision/supersession gate drifted.",
    )
    for execution_id, row in expected.items():
        _require(
            {
                key: observed[execution_id].get(key)
                for key in row
            }
            == row,
            f"Collision row drifted for {execution_id}.",
        )
    bindings = _mapping(
        payload.get("bindings"), label="collision evidence bindings"
    )
    for label, binding in bindings.items():
        row = _mapping(binding, label=f"{label} collision binding")
        path = repo_root / safe_relative_path(
            row.get("path"), label=f"{label} collision path"
        )
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size
            == int(row.get("size_bytes", -1))
            and sha256_file(path) == row.get("sha256"),
            f"{label} collision evidence drifted.",
        )


def _validate_source_audit(
    *,
    payload: Mapping[str, Any],
    resume_ids: set[str],
    fresh_ids: set[str],
) -> None:
    verify_self_digest(payload, label="source-lock audit")
    sweep = _mapping(payload.get("sweep"), label="audit sweep")
    anchor = _mapping(payload.get("anchor"), label="audit anchor")
    rows = _sequence(
        payload.get("planned_rows"), label="audit planned rows"
    )
    by_id = {
        str(row["execution_id"]): row
        for row in rows
        if isinstance(row, Mapping)
    }
    _require(
        payload.get("schema") == "source_locked_sensitivity_audit_v1"
        and payload.get("package_id") == PACKAGE_ID
        and sweep.get("variable") == "maximum_controller_rounds"
        and sweep.get("grid") == [TARGET_HORIZON]
        and sweep.get("settings_changed")
        == ["maximum_controller_rounds"]
        and sweep.get("unresolved_source_fields") == []
        and sweep.get("fields_added_by_current_defaults") == []
        and int(
            anchor.get("authenticated_resume_anchor_count", -1)
        )
        == RESUME_COUNT
        and int(anchor.get("blocked_fresh_anchor_count", -1))
        == FRESH_COUNT
        and anchor.get("all_available_resume_anchors_close") is True
        and anchor.get("all_fresh_rows_blocked") is True
        and set(by_id) == resume_ids.union(fresh_ids),
        "Source-lock audit header drifted.",
    )
    for execution_id, row in by_id.items():
        source_anchor = _mapping(
            row.get("anchor"), label=f"{execution_id} audit anchor"
        )
        expected_resume = execution_id in resume_ids
        _require(
            row.get("changed_fields_vs_source")
            == ["maximum_controller_rounds"]
            and row.get("non_swept_settings_diff") == []
            and row.get("fields_added_by_current_defaults") == []
            and row.get("unresolved_source_fields") == []
            and source_anchor.get("non_swept_settings_diff") == []
            and source_anchor.get("anchor_reproduces_source")
            is expected_resume
            and (
                row.get("status")
                == "passed_authenticated_resume_anchor"
            )
            is expected_resume
            and (
                row.get("status")
                == "blocked_live_r50_predecessor"
            )
            is (not expected_resume),
            f"{execution_id} source-lock audit row drifted.",
        )


def validate_package(
    *, full_archive_scan: bool = True
) -> dict[str, Any]:
    repo_root = repo_root_from_script(__file__)
    _require(
        not (PACKAGE_DIR / "submit.sub").exists()
        and not (PACKAGE_DIR / "authority").exists(),
        "The collision-blocked package gained submit or authority state.",
    )
    manifest = load_json(
        PACKAGE_DIR / PACKAGE_MANIFEST_NAME,
        label="package manifest",
    )
    verify_self_digest(manifest, label="package manifest")
    _require(
        manifest.get("schema") == PACKAGE_MANIFEST_SCHEMA
        and manifest.get("package_id") == PACKAGE_ID
        and manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("status")
        == "passed_inert_collision_blocked"
        and int(manifest.get("cell_count", -1)) == CELL_COUNT
        and int(manifest.get("authenticated_resume_count", -1))
        == RESUME_COUNT
        and int(manifest.get("fresh_count", -1)) == FRESH_COUNT
        and int(manifest.get("source_horizon", -1))
        == SOURCE_HORIZON
        and int(manifest.get("target_horizon", -1))
        == TARGET_HORIZON
        and manifest.get("active_gradient_policy")
        == ACTIVE_GRADIENT_POLICY
        and manifest.get("resource_weighting_scope")
        == RESOURCE_WEIGHTING_SCOPE
        and manifest.get("submit_descriptor_present") is False
        and manifest.get("authority_overlay_present") is False
        and manifest.get("execution_authorized") is False
        and manifest.get("submission_authorized") is False
        and manifest.get("submission_ready") is False
        and manifest.get("submitted") is False
        and manifest.get("remote_stage") is False
        and manifest.get("condor_submit") is False,
        "Package-manifest inert identity drifted.",
    )

    provenance = load_json(
        repo_root / SOURCE_REPORT_RELATIVE,
        label="stationary-core provenance",
    )
    rows = planned_rows(repo_root=repo_root, provenance=provenance)
    row_by_id = {str(row["execution_id"]): row for row in rows}
    resume_ids = {
        str(row["execution_id"])
        for row in rows
        if row["execution_mode"] == "authenticated_resume_50_to_70"
    }
    fresh_ids = set(row_by_id).difference(resume_ids)
    _require(
        len(row_by_id) == CELL_COUNT
        and len(resume_ids) == RESUME_COUNT
        and len(fresh_ids) == FRESH_COUNT,
        "Reconstructed RA36 matrix drifted.",
    )

    document_names = {
        "source_archives": SOURCE_ARCHIVES_NAME,
        "resume_inputs": RESUME_INPUTS_NAME,
        "source_lock_audit": SOURCE_LOCK_AUDIT_NAME,
        "collision_status": COLLISION_STATUS_NAME,
        "execution_plan": EXECUTION_PLAN_NAME,
    }
    documents: dict[str, dict[str, Any]] = {}
    for key, name in document_names.items():
        binding = _mapping(
            manifest.get(key), label=f"{key} manifest binding"
        )
        _require(
            binding.get("path") == name,
            f"{key} manifest path drifted.",
        )
        path = PACKAGE_DIR / name
        payload = load_json(path, label=key)
        digest = verify_self_digest(payload, label=key)
        _require(
            binding.get("sha256") == digest
            and binding.get("file_sha256") == sha256_file(path),
            f"{key} manifest binding drifted.",
        )
        documents[key] = payload

    _validate_source_archives(
        payload=documents["source_archives"],
        repo_root=repo_root,
        full_archive_scan=full_archive_scan,
    )
    _validate_resume_inputs(
        payload=documents["resume_inputs"],
        resume_ids=resume_ids,
        full_archive_scan=full_archive_scan,
    )
    _validate_collision(
        payload=documents["collision_status"],
        repo_root=repo_root,
        fresh_ids={
            str(row["base_execution_id"])
            for row in rows
            if row["execution_id"] in fresh_ids
        },
    )
    _validate_source_audit(
        payload=documents["source_lock_audit"],
        resume_ids=resume_ids,
        fresh_ids=fresh_ids,
    )

    plan = documents["execution_plan"]
    verify_self_digest(plan, label="execution plan")
    _require(
        plan.get("schema") == EXECUTION_PLAN_SCHEMA
        and plan.get("package_id") == PACKAGE_ID
        and plan.get("campaign_id") == CAMPAIGN_ID
        and plan.get("execution_ids") == list(row_by_id)
        and plan.get("only_scientific_change")
        == "maximum_controller_rounds_50_to_70"
        and plan.get("execution_authorized") is False
        and plan.get("submission_authorized") is False
        and plan.get("submission_ready") is False
        and plan.get("submitted") is False
        and plan.get("remote_stage") is False
        and plan.get("condor_submit") is False
        and "live_r50_predecessors_9397758_0_through_8"
        in plan.get("submission_blockers", []),
        "Execution plan is no longer inert or source-locked.",
    )

    source_families = _mapping(
        documents["source_archives"]["families"],
        label="source families",
    )
    resume_cells = _mapping(
        documents["resume_inputs"]["cells"],
        label="resume cells",
    )
    job_bindings = _sequence(
        manifest.get("jobs"), label="job bindings"
    )
    jobs_by_id = {
        str(item["execution_id"]): item
        for item in job_bindings
        if isinstance(item, Mapping)
    }
    _require(
        len(job_bindings) == CELL_COUNT
        and set(jobs_by_id) == set(row_by_id),
        "Job-manifest matrix drifted.",
    )
    for execution_id, row in row_by_id.items():
        binding = _mapping(
            jobs_by_id[execution_id],
            label=f"{execution_id} job binding",
        )
        path = _package_path(
            binding.get("path"), label=f"{execution_id} job path"
        )
        _require(
            path
            == PACKAGE_DIR / "jobs" / f"{execution_id}.json"
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size
            == int(binding.get("size_bytes", -1))
            and sha256_file(path) == binding.get("sha256"),
            f"{execution_id} job file binding drifted.",
        )
        job = load_json(path, label=f"{execution_id} job")
        canonical_digest = verify_self_digest(
            job, label=f"{execution_id} job"
        )
        for key, value in row.items():
            _require(
                job.get(key) == value,
                f"{execution_id} job row field drifted: {key}.",
            )
        family = _mapping(
            source_families[row["source_family"]],
            label=f"{execution_id} source family",
        )
        expected_resume = resume_cells.get(execution_id)
        _require(
            job.get("schema") == JOB_SCHEMA
            and job.get("package_id") == PACKAGE_ID
            and job.get("campaign_id") == CAMPAIGN_ID
            and binding.get("canonical_sha256")
            == canonical_digest
            and job.get("source_archive")
            == family.get("packaged_archive")
            and job.get("source_archive_manifest")
            == family.get("packaged_manifest")
            and (
                job.get("resume_input") is not None
            )
            is (expected_resume is not None)
            and job.get("source_lock_delta")
            == {
                "variable": "maximum_controller_rounds",
                "from": SOURCE_HORIZON,
                "to": TARGET_HORIZON,
                "changed_fields_vs_source": [
                    "maximum_controller_rounds"
                ],
                "non_swept_settings_diff": [],
            }
            and job.get("authorization_schema")
            == AUTHORIZATION_SCHEMA
            and job.get("global_submission_blocked") is True
            and job.get("submission_ready") is False
            and job.get("execution_authorized") is False
            and job.get("submission_authorized") is False
            and job.get("submitted") is False,
            f"{execution_id} job closure drifted.",
        )
        if expected_resume is not None:
            resume = _mapping(
                job["resume_input"],
                label=f"{execution_id} job resume",
            )
            _require(
                resume.get("archive")
                == expected_resume.get("archive")
                and resume.get("checkpoint_path")
                == expected_resume.get("checkpoint_path")
                and resume.get("checkpoint_sha256")
                == expected_resume.get("checkpoint_sha256")
                and resume.get("member_count") == 3
                and resume.get("pointer_closed") is True,
                f"{execution_id} job resume binding drifted.",
            )
        else:
            _require(
                row.get("collision_status")
                == "blocked_live_r50_predecessor"
                and row["collision"]["cluster_id"]
                == COLLISION_CLUSTER_ID
                and row["collision"]["proc_id"]
                in COLLISION_PROC_IDS,
                f"{execution_id} fresh collision gate drifted.",
            )

    queue_binding = _mapping(
        manifest.get("queue"), label="queue binding"
    )
    queue_path = PACKAGE_DIR / QUEUE_NAME
    queue_lines = queue_path.read_text(encoding="utf-8").splitlines()
    _require(
        queue_binding.get("path") == QUEUE_NAME
        and queue_binding.get("kind")
        == "inert_planning_queue_not_condor_queue"
        and int(queue_binding.get("row_count", -1)) == CELL_COUNT
        and queue_binding.get("sha256") == sha256_file(queue_path)
        and len(queue_lines) == CELL_COUNT
        and all(len(line.split("\t")) == 8 for line in queue_lines),
        "Inert planning queue drifted.",
    )
    for line, row in zip(queue_lines, rows):
        fields = line.split("\t")
        _require(
            fields[:4]
            == [
                row["execution_id"],
                row["execution_mode"],
                row["collision_status"],
                row["source_family"],
            ],
            f"Planning queue row drifted for {row['execution_id']}.",
        )

    control_bindings = {
        str(item["path"]): item
        for item in _sequence(
            manifest.get("control_files"),
            label="control-file bindings",
        )
        if isinstance(item, Mapping)
    }
    _require(
        set(control_bindings) == set(CONTROL_FILES),
        "Control-file closure drifted.",
    )
    for relative in CONTROL_FILES:
        path = PACKAGE_DIR / relative
        binding = control_bindings[relative]
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size
            == int(binding.get("size_bytes", -1))
            and sha256_file(path) == binding.get("sha256"),
            f"Control file drifted: {relative}.",
        )

    return {
        "status": "passed_inert_collision_blocked",
        "package_id": PACKAGE_ID,
        "cell_count": CELL_COUNT,
        "authenticated_resume_count": RESUME_COUNT,
        "fresh_count": FRESH_COUNT,
        "full_archive_scan": full_archive_scan,
        "collision_cluster_id": COLLISION_CLUSTER_ID,
        "collision_proc_ids": list(COLLISION_PROC_IDS),
        "submission_ready": False,
        "execution_authorized": False,
        "submission_authorized": False,
        "submitted": False,
        "package_manifest_sha256": manifest["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip compressed member rehashing for the large resume inputs.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = validate_package(
            full_archive_scan=not args.metadata_only
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
