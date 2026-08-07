"""Contracts for the explicit checkpoint-retention operational overlay."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    PACKAGE_ID,
    SOURCE_HORIZON,
    TARGET_HORIZON,
    PackageContractError,
    canonical_json_bytes,
    canonical_sha256,
    digested,
    safe_relative_path,
)


OVERLAY_ID = (
    "paper_i_ra_adapt_stationary_core_ra36_r70_"
    "continuation_operational_overlay_v2"
)
OVERLAY_PACKAGE_ID = f"{PACKAGE_ID}_operational_overlay_v2"

CHECKPOINT_MEMBER = "pipelines/static_adapt/current_checkpoint.py"
PARENT_CHECKPOINT_SHA256 = (
    "16ffddfdbf20674c50af7b797131efa40478c5281d16f4f034d7db49b8249cb8"
)
REPAIRED_CHECKPOINT_SHA256 = (
    "87e032010e009261de415101b717ff38fdb3d9b894b18d1939e6b219d94219f3"
)

EFFECTIVE_CORE_FAMILY = "stationary_core_v11_retention_v2"
EFFECTIVE_ALWAYS_FAMILY = "always_factorial_retention_v2"
EFFECTIVE_FAMILY_BY_BASE_FAMILY = {
    "stationary_core_v11": EFFECTIVE_CORE_FAMILY,
    "always_factorial_v1": EFFECTIVE_ALWAYS_FAMILY,
    "always_factorial_v2": EFFECTIVE_ALWAYS_FAMILY,
}

EFFECTIVE_SOURCES_NAME = "effective_source_archives_v2.json"
COLLISION_EVIDENCE_NAME = "collision_evidence_v2.json"
SOURCE_LOCK_AUDIT_V2_NAME = "source_lock_audit_v2.json"
EXECUTION_PLAN_V2_NAME = "execution_plan_v2.json"
QUEUE_V2_NAME = "queue_v2.tsv"
BUILD_RECEIPT_V2_NAME = "operational_overlay_v2_build_receipt.json"
OVERLAY_MANIFEST_NAME = "operational_overlay_v2_manifest.json"
JOBS_V2_DIR = "jobs_v2"
EFFECTIVE_SOURCES_DIR = "effective_source_archives"

OVERLAY_CONTROL_FILES = (
    "operational_overlay_v2_contract.py",
    "build_operational_overlay_v2.py",
    "validate_operational_overlay_v2.py",
    "run_cell_v2.py",
)
OVERLAY_GENERATED_FILES = (
    EFFECTIVE_SOURCES_NAME,
    COLLISION_EVIDENCE_NAME,
    SOURCE_LOCK_AUDIT_V2_NAME,
    EXECUTION_PLAN_V2_NAME,
    QUEUE_V2_NAME,
    BUILD_RECEIPT_V2_NAME,
    OVERLAY_MANIFEST_NAME,
)

EFFECTIVE_SOURCES_SCHEMA = (
    "paper_i_ra_adapt_r70_effective_source_archives_v2"
)
COLLISION_EVIDENCE_SCHEMA = (
    "paper_i_ra_adapt_r70_collision_evidence_v2"
)
SOURCE_LOCK_AUDIT_V2_SCHEMA = "source_locked_sensitivity_audit_v2"
EXECUTION_PLAN_V2_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_execution_plan_v2"
)
JOB_V2_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_job_v2"
)
SCIENTIFIC_SETTINGS_SCHEMA = (
    "paper_i_ra_adapt_r70_scientific_settings_v2"
)
OPERATIONAL_SETTINGS_SCHEMA = (
    "paper_i_ra_adapt_r70_operational_settings_v2"
)
EFFECTIVE_EXECUTION_CONTRACT_SCHEMA = (
    "paper_i_ra_adapt_r70_effective_execution_contract_v2"
)
AUTHORIZATION_V2_SCHEMA = (
    "paper_i_ra_adapt_r70_execution_authorization_v2"
)
COLLISION_CLEARANCE_SCHEMA = (
    "paper_i_ra_adapt_r70_fresh_collision_clearance_v2"
)
BUILD_RECEIPT_V2_SCHEMA = (
    "paper_i_ra_adapt_r70_operational_overlay_build_receipt_v2"
)
OVERLAY_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_r70_operational_overlay_manifest_v2"
)

DERIVED_PROTOCOL_CHANGED_PATHS = (
    "horizon",
    "request.execution.stop.maximum_controller_rounds",
    "sha256",
    "stopping_rule.maximum_controller_rounds",
)
LOGICAL_OUTPUT_FILENAMES = {
    "checkpoint": "checkpoint.json",
    "estimator_ledger": "estimator_ledger.json",
    "execution_manifest": "execution_manifest.json",
    "result": "result.json",
    "summary": "summary.json",
}


def canonical_file_binding(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the path/hash/size projection used in runtime contracts."""

    result = {
        "path": safe_relative_path(
            binding.get("path"), label="file binding path"
        ).as_posix(),
        "sha256": str(binding.get("sha256", "")),
        "size_bytes": int(binding.get("size_bytes", -1)),
    }
    canonical = binding.get("canonical_sha256")
    if canonical is not None:
        result["canonical_sha256"] = str(canonical)
    if (
        len(result["sha256"]) != 64
        or result["size_bytes"] < 0
        or (
            canonical is not None
            and len(str(canonical)) != 64
        )
    ):
        raise PackageContractError("File binding is incomplete.")
    return result


def _resume_policy(job: Mapping[str, Any]) -> dict[str, Any]:
    resume = job.get("resume_input")
    if resume is None:
        collision = job.get("collision")
        if not isinstance(collision, Mapping):
            raise PackageContractError(
                "Fresh row lacks its collision predecessor binding."
            )
        return {
            "kind": "fresh_start",
            "checkpoint_consumed": False,
            "collision_clearance_required": True,
            "predecessor": {
                "cluster_id": int(collision["cluster_id"]),
                "proc_id": int(collision["proc_id"]),
                "source_execution_id": str(
                    collision["source_execution_id"]
                ),
            },
        }
    if not isinstance(resume, Mapping):
        raise PackageContractError("Resume input must be a mapping.")
    members = resume.get("members")
    if not isinstance(members, list):
        raise PackageContractError(
            "Resume input lacks its member bindings."
        )
    by_role = {
        str(row["role"]): canonical_file_binding(row)
        for row in members
        if isinstance(row, Mapping)
    }
    if set(by_role) != {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }:
        raise PackageContractError(
            "Resume input is not a pointer-closed triplet."
        )
    return {
        "kind": "accepted_state_resume",
        "checkpoint_consumed": True,
        "source_horizon": SOURCE_HORIZON,
        "archive": canonical_file_binding(
            _mapping(resume.get("archive"), label="resume archive")
        ),
        "checkpoint_path": safe_relative_path(
            resume.get("checkpoint_path"),
            label="resume checkpoint path",
        ).as_posix(),
        "checkpoint_sha256": str(
            resume.get("checkpoint_sha256", "")
        ),
        "members_by_role": by_role,
        "pointer_closed": bool(resume.get("pointer_closed")),
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PackageContractError(f"{label} must be a mapping.")
    return value


def build_effective_execution_contract(
    *,
    job: Mapping[str, Any],
    derived_protocol_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete scientific/operational contract for one row."""

    derived = json.loads(
        canonical_json_bytes(dict(derived_protocol_payload))
    )
    derived_sha256 = str(derived.get("sha256", ""))
    if (
        len(derived_sha256) != 64
        or int(derived.get("horizon", -1)) != TARGET_HORIZON
        or int(
            derived.get("request", {})
            .get("execution", {})
            .get("stop", {})
            .get("maximum_controller_rounds", -1)
        )
        != TARGET_HORIZON
    ):
        raise PackageContractError(
            "Derived protocol payload is not the bound r70 protocol."
        )
    source_protocol = _mapping(
        job.get("source_protocol"), label="source protocol"
    )
    scientific_settings = {
        "schema": SCIENTIFIC_SETTINGS_SCHEMA,
        "execution_id": str(job["execution_id"]),
        "source_protocol": {
            key: source_protocol[key]
            for key in (
                "path",
                "sha256",
                "canonical_sha256",
                "size_bytes",
                "route_profile",
                "route_contract_sha256",
            )
        },
        "derived_protocol_payload": derived,
        "derived_protocol_sha256": derived_sha256,
        "changed_protocol_paths": list(
            DERIVED_PROTOCOL_CHANGED_PATHS
        ),
        "only_scientific_change": (
            "maximum_controller_rounds_50_to_70"
        ),
        "source_horizon": SOURCE_HORIZON,
        "target_horizon": TARGET_HORIZON,
        "stationary_gradient_policy": str(
            job["active_gradient_policy"]
        ),
        "resource_weighting_scope": str(
            job["resource_weighting_scope"]
        ),
        "non_swept_settings_diff": [],
        "fields_added_by_current_defaults": [],
    }
    scientific_settings_sha256 = canonical_sha256(
        scientific_settings
    )

    root = safe_relative_path(
        job.get("expected_output_root"),
        label="logical output root",
    ).as_posix()
    logical_outputs = {
        role: f"{root}/{filename}"
        for role, filename in sorted(
            LOGICAL_OUTPUT_FILENAMES.items()
        )
    }
    operational_settings = {
        "schema": OPERATIONAL_SETTINGS_SCHEMA,
        "execution_id": str(job["execution_id"]),
        "execution_mode": str(job["execution_mode"]),
        "effective_source": {
            "family": str(job["effective_source_family"]),
            "archive": canonical_file_binding(
                _mapping(
                    job.get("effective_source_archive"),
                    label="effective source archive",
                )
            ),
            "manifest": canonical_file_binding(
                _mapping(
                    job.get("effective_source_archive_manifest"),
                    label="effective source manifest",
                )
            ),
            "delta_receipt": canonical_file_binding(
                _mapping(
                    job.get("effective_source_delta_receipt"),
                    label="effective source delta receipt",
                )
            ),
        },
        "resume_policy": _resume_policy(job),
        "operational_controls": {
            "maximum_controller_rounds": TARGET_HORIZON,
            "observation": {
                "checkpoint": {
                    "logical_path": logical_outputs[
                        "checkpoint"
                    ],
                    "every_controller_rounds": 1,
                    "keep_history_tail": 100,
                },
                "estimator_ledger": {
                    "logical_path": logical_outputs[
                        "estimator_ledger"
                    ]
                },
                "resource_rounds": None,
            },
        },
        "logical_output_template": {
            "root": root,
            "files": logical_outputs,
        },
        "resources": json.loads(
            canonical_json_bytes(
                _mapping(job.get("resources"), label="resources")
            )
        ),
        "collision_status": str(job["collision_status"]),
    }
    operational_settings_sha256 = canonical_sha256(
        operational_settings
    )
    return digested(
        {
            "schema": EFFECTIVE_EXECUTION_CONTRACT_SCHEMA,
            "overlay_id": OVERLAY_ID,
            "package_id": OVERLAY_PACKAGE_ID,
            "execution_id": str(job["execution_id"]),
            "scientific_settings": scientific_settings,
            "scientific_settings_sha256": (
                scientific_settings_sha256
            ),
            "operational_settings": operational_settings,
            "operational_settings_sha256": (
                operational_settings_sha256
            ),
        }
    )


def normalized_protocol_without_horizon(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only the four expected source→r70 delta paths."""

    result = json.loads(canonical_json_bytes(dict(payload)))
    replacements = {
        "horizon": "<r50-or-r70>",
        "request.execution.stop.maximum_controller_rounds": (
            "<r50-or-r70>"
        ),
        "sha256": "<source-or-derived-digest>",
        "stopping_rule.maximum_controller_rounds": (
            "<r50-or-r70>"
        ),
    }
    for path, replacement in replacements.items():
        cursor: Any = result
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise PackageContractError(
                    f"Protocol lacks normalized path: {path}"
                )
            cursor = cursor[part]
        if (
            not isinstance(cursor, dict)
            or parts[-1] not in cursor
        ):
            raise PackageContractError(
                f"Protocol lacks normalized path: {path}"
            )
        cursor[parts[-1]] = replacement
    return result


def effective_contract_sha256(
    contract: Mapping[str, Any]
) -> str:
    unsigned = dict(contract)
    observed = str(unsigned.pop("sha256", ""))
    expected = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    if observed != expected:
        raise PackageContractError(
            "Effective execution contract self digest drifted."
        )
    return observed
