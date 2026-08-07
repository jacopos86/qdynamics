#!/usr/bin/env python3
"""Fail-closed contract for the Paper-I stationary-core RA r50->r70 scaffold.

This sibling does not reinterpret or mutate the sealed v1/overlay-v2 package.
It binds that package byte-for-byte, projects its 36 scientific contracts, and
changes the intended execution shape to 36 authenticated resumes.  Twenty-seven
resume archives are reused read-only.  The remaining nine rows stay blocked
until exact predecessor and pointer-closed resume bindings are supplied.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import ijson


sys.dont_write_bytecode = True


class ScaffoldContractError(ValueError):
    """Raised when the sealed parent or resume scaffold drifts."""


CONTROLLED_CYCLE_ROOT = Path(__file__).resolve().parent.parent
CONTROLLED_CYCLE_VALIDATOR_PATH = (
    CONTROLLED_CYCLE_ROOT / "validate_controlled_cycle_archive.py"
)
CONTROLLED_CYCLE_VALIDATOR_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "validate_controlled_cycle_archive.py"
)
CONTROLLED_CYCLE_VALIDATOR_SHA256 = (
    "c7e5cac3f1b9ceba29c34fc9d49e8ae1f56154d5d92e4006d1a824e172ed10b9"
)
CONTROLLED_CYCLE_VALIDATOR_SIZE_BYTES = 39399


def verify_controlled_cycle_dependency(
    path: Path = CONTROLLED_CYCLE_VALIDATOR_PATH,
) -> dict[str, Any]:
    """Verify attempt-authentication code before importing or invoking it."""

    candidate = Path(path)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.stat().st_size != CONTROLLED_CYCLE_VALIDATOR_SIZE_BYTES
    ):
        raise ScaffoldContractError(
            "Controlled-cycle validator dependency is missing or drifted."
        )
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != CONTROLLED_CYCLE_VALIDATOR_SHA256:
        raise ScaffoldContractError(
            "Controlled-cycle validator dependency bytes drifted."
        )
    return {
        "path": CONTROLLED_CYCLE_VALIDATOR_RELATIVE_PATH,
        "sha256": digest,
        "size_bytes": CONTROLLED_CYCLE_VALIDATOR_SIZE_BYTES,
        "role": "predecessor_attempt_authentication_validator",
    }


CONTROLLED_CYCLE_VALIDATOR_BINDING = verify_controlled_cycle_dependency()
if str(CONTROLLED_CYCLE_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROLLED_CYCLE_ROOT))

from validate_controlled_cycle_archive import (  # noqa: E402
    COMPLETION_RECEIPT_SCHEMA as CONTROLLED_COMPLETION_RECEIPT_SCHEMA,
    ControlledCycleArchiveError,
    ExpectedAttempt,
    _load_json_file as load_controlled_json_file,
    validate_attempt_archive,
)

PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_ra36_r70_"
    "continuation_20260731_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_stationary_core_r70_continuation_v2"
)
RUN_CLASS = "paper_facing"
EXECUTION_TARGET = "chtc"
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
CELL_COUNT = 36
INHERITED_RESUME_COUNT = 27
PENDING_RESUME_COUNT = 9
TARGET_RESUME_COUNT = 36

PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_ra36_r70_continuation_20260731_v2_chtc"
)
EXTERNAL_EVIDENCE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_ra36_r70_continuation_20260731_input_evidence_v1"
)
SEALED_PARENT_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "stationary_core_ra36_r70_continuation_20260731_v1_chtc"
)
SEALED_PARENT_MANIFEST_NAME = "operational_overlay_v2_manifest.json"
SEALED_PARENT_MANIFEST_CANONICAL_SHA256 = (
    "feacd78fb7e2895b7eb8a1a97793378a19ac80d34a47629da00b57c3673b5688"
)
SEALED_PARENT_MANIFEST_FILE_SHA256 = (
    "ec7d578c313b554a6b98e642a252088b820b341cfbbf59763f6ca137a61b1ef5"
)
SEALED_PARENT_PACKAGE_ID = (
    "paper_i_ra_adapt_stationary_core_ra36_r70_"
    "continuation_20260731_v1_chtc_operational_overlay_v2"
)

ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
ONLY_SCIENTIFIC_CHANGE = "maximum_controller_rounds_50_to_70"
EXECUTION_MODE = "authenticated_resume_50_to_70"

SCIENTIFIC_SETTINGS_SCHEMA = (
    "paper_i_ra_adapt_r70_scientific_settings_v2"
)
SCAFFOLD_JOB_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_resume_scaffold_job_v2"
)
PREDECESSOR_REQUIREMENT_SCHEMA = (
    "paper_i_ra_adapt_r70_predecessor_requirement_v2"
)
PREDECESSOR_BINDING_SCHEMA = (
    "paper_i_ra_adapt_r70_predecessor_binding_v2"
)
SCHEDULER_TERMINAL_RECEIPT_SCHEMA = (
    "paper_i_condor_exact_proc_terminal_receipt_v1"
)
PACKAGE_PLAN_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_resume_plan_v2"
)
TRANSFER_PLAN_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_row_transfer_plan_v2"
)
SCAFFOLD_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_scaffold_manifest_v2"
)
ACTIVATION_INPUT_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_activation_inputs_v2"
)
ACTIVATION_AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_execution_authorization_v2"
)
RUNTIME_BUNDLE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_runtime_bundle_manifest_v2"
)
IMAGE_VERIFICATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_image_verification_v2"
)
RESOURCE_EVIDENCE_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_resource_evidence_v2"
)
RESOURCE_OBSERVATION_SCHEMA = (
    "paper_i_ra_adapt_stationary_core_r70_resource_observation_v2"
)

JOBS_DIR = "jobs"
PREDECESSOR_PLACEHOLDERS_DIR = "predecessor_placeholders"
PREDECESSOR_BINDINGS_DIR = "predecessor_bindings"
NEW_RESUME_INPUTS_DIR = "resume_inputs"
RUNTIME_BUNDLE_RELATIVE = "runtime/ra_r70_row_runtime.tar.gz"
PACKAGE_PLAN_NAME = "package_plan.json"
TRANSFER_PLAN_NAME = "row_transfer_plan.json"
TRANSFER_QUEUE_NAME = "row_transfer_plan.tsv"
PREDECESSOR_REQUIREMENTS_NAME = "predecessor_requirements.json"
SCAFFOLD_MANIFEST_NAME = "scaffold_manifest.json"

CONTROL_FILES = (
    "scaffold_contract.py",
    "materialize_scaffold.py",
    "validate_scaffold.py",
    "build_activation.py",
    "README.md",
)
GENERATED_FILES = (
    PACKAGE_PLAN_NAME,
    TRANSFER_PLAN_NAME,
    TRANSFER_QUEUE_NAME,
    PREDECESSOR_REQUIREMENTS_NAME,
    SCAFFOLD_MANIFEST_NAME,
)

REGIME_CUTOFF_PAIRS = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ROUTE_IDS = (
    "ra_macro_append_only",
    "ra_macro_plateau",
    "ra_macro_always",
    "ra_singleton_append_only",
    "ra_singleton_plateau",
    "ra_singleton_always",
)

# These are exact scheduler predecessors, not a range-shaped permission.
PENDING_PREDECESSORS = {
    "core__intermediate_weak__nph3__ra_macro_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 0,
        "source_execution_id": (
            "core__intermediate_weak__nph3__ra_macro_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__strong_weak_u8__nph3__ra_macro_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 1,
        "source_execution_id": (
            "core__strong_weak_u8__nph3__ra_macro_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__strong_weak_u8__nph3__ra_singleton_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 2,
        "source_execution_id": (
            "core__strong_weak_u8__nph3__ra_singleton_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__weak_strong__nph7__ra_macro_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 3,
        "source_execution_id": (
            "core__weak_strong__nph7__ra_macro_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__weak_strong__nph7__ra_singleton_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 4,
        "source_execution_id": (
            "core__weak_strong__nph7__ra_singleton_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__intermediate_strong__nph7__ra_macro_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 5,
        "source_execution_id": (
            "core__intermediate_strong__nph7__ra_macro_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__intermediate_strong__nph7__ra_singleton_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 6,
        "source_execution_id": (
            "core__intermediate_strong__nph7__ra_singleton_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__strong_strong_u8__nph7__ra_macro_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 7,
        "source_execution_id": (
            "core__strong_strong_u8__nph7__ra_macro_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
    "core__strong_strong_u8__nph7__ra_singleton_always__r70": {
        "cluster_id": 9397758,
        "proc_id": 8,
        "source_execution_id": (
            "core__strong_strong_u8__nph7__ra_singleton_always"
            "__gradient_stationary__phase1_cost_off"
        ),
    },
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def repo_root_from_script(script_path: str | Path) -> Path:
    return Path(script_path).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise ScaffoldContractError(f"{label} is missing or unsafe: {path}")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScaffoldContractError(f"{label} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ScaffoldContractError(f"{label} must be a JSON object.")
    return payload


def verify_self_digest(payload: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop("sha256", None)
    if not isinstance(observed, str) or SHA256_RE.fullmatch(observed) is None:
        raise ScaffoldContractError(f"{label} lacks a lowercase SHA-256.")
    if canonical_sha256(unsigned) != observed:
        raise ScaffoldContractError(f"{label} self digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ScaffoldContractError(f"{label} must be a non-empty path.")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ScaffoldContractError(f"{label} is not a safe relative path.")
    return candidate


def repo_file_binding(path: Path, *, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ScaffoldContractError("Bound file escaped the repository.") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ScaffoldContractError(f"Bound file is missing or unsafe: {relative}")
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def expected_execution_ids() -> set[str]:
    return {
        f"core__{regime}__nph{nph}__{route}__r70"
        for regime, nph in REGIME_CUTOFF_PAIRS
        for route in ROUTE_IDS
    }


def parent_root(repo_root: Path) -> Path:
    return repo_root / SEALED_PARENT_RELATIVE_ROOT


def package_root(repo_root: Path) -> Path:
    return repo_root / PACKAGE_RELATIVE_ROOT


def verify_exact_binding(
    path: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
    rehash: bool = True,
) -> None:
    if not path.is_file() or path.is_symlink():
        raise ScaffoldContractError(f"{label} is missing or unsafe.")
    if path.stat().st_size != int(binding.get("size_bytes", -1)):
        raise ScaffoldContractError(f"{label} size drifted.")
    expected_sha = binding.get("sha256")
    if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
        raise ScaffoldContractError(f"{label} binding lacks a SHA-256.")
    if rehash and sha256_file(path) != expected_sha:
        raise ScaffoldContractError(f"{label} bytes drifted.")


def load_sealed_parent_jobs(
    repo_root: Path,
    *,
    rehash_jobs: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load only the exact sealed byte surface; never rederive semantics."""

    root = parent_root(repo_root)
    manifest_path = root / SEALED_PARENT_MANIFEST_NAME
    if sha256_file(manifest_path) != SEALED_PARENT_MANIFEST_FILE_SHA256:
        raise ScaffoldContractError("Sealed parent manifest bytes drifted.")
    manifest = load_json(manifest_path, label="sealed parent manifest")
    if verify_self_digest(manifest, label="sealed parent manifest") != (
        SEALED_PARENT_MANIFEST_CANONICAL_SHA256
    ):
        raise ScaffoldContractError("Sealed parent manifest identity drifted.")
    if (
        manifest.get("package_id") != SEALED_PARENT_PACKAGE_ID
        or manifest.get("cell_count") != CELL_COUNT
        or manifest.get("authenticated_resume_count") != INHERITED_RESUME_COUNT
        or manifest.get("fresh_count") != PENDING_RESUME_COUNT
        or manifest.get("submission_ready") is not False
        or manifest.get("submitted") is not False
    ):
        raise ScaffoldContractError("Sealed parent manifest contract drifted.")
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != CELL_COUNT:
        raise ScaffoldContractError("Sealed parent job inventory is incomplete.")
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_binding in enumerate(raw_jobs):
        if not isinstance(raw_binding, Mapping):
            raise ScaffoldContractError(f"Parent job binding {index} is malformed.")
        execution_id = str(raw_binding.get("execution_id", ""))
        relative = safe_relative_path(
            raw_binding.get("path"), label=f"{execution_id} parent job path"
        )
        path = root / relative
        verify_exact_binding(
            path,
            raw_binding,
            label=f"{execution_id} sealed parent job",
            rehash=rehash_jobs,
        )
        job = load_json(path, label=f"{execution_id} sealed parent job")
        canonical = verify_self_digest(job, label=f"{execution_id} sealed parent job")
        if (
            execution_id != job.get("execution_id")
            or raw_binding.get("canonical_sha256") != canonical
            or execution_id in seen
        ):
            raise ScaffoldContractError(f"{execution_id} parent job identity drifted.")
        seen.add(execution_id)
        jobs.append(job)
    if seen != expected_execution_ids():
        raise ScaffoldContractError("Sealed parent is not the exact 6x6 core matrix.")
    return manifest, jobs


def validate_scientific_projection(job: Mapping[str, Any]) -> dict[str, Any]:
    contract = job.get("effective_execution_contract")
    if not isinstance(contract, Mapping):
        raise ScaffoldContractError("Parent job lacks an effective contract.")
    scientific = contract.get("scientific_settings")
    if not isinstance(scientific, Mapping):
        raise ScaffoldContractError("Parent job lacks scientific settings.")
    digest = canonical_sha256(scientific)
    if (
        scientific.get("schema") != SCIENTIFIC_SETTINGS_SCHEMA
        or scientific.get("execution_id") != job.get("execution_id")
        or scientific.get("source_horizon") != SOURCE_HORIZON
        or scientific.get("target_horizon") != TARGET_HORIZON
        or scientific.get("only_scientific_change") != ONLY_SCIENTIFIC_CHANGE
        or scientific.get("stationary_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or scientific.get("resource_weighting_scope") != RESOURCE_WEIGHTING_SCOPE
        or scientific.get("non_swept_settings_diff") != []
        or scientific.get("fields_added_by_current_defaults") != []
        or contract.get("scientific_settings_sha256") != digest
        or job.get("scientific_settings_sha256") != digest
    ):
        raise ScaffoldContractError(
            f"{job.get('execution_id')} scientific projection drifted."
        )
    derived = scientific.get("derived_protocol_payload")
    source_protocol = scientific.get("source_protocol")
    if (
        not isinstance(derived, Mapping)
        or not isinstance(source_protocol, Mapping)
        or derived.get("sha256") != scientific.get("derived_protocol_sha256")
        or int(derived.get("horizon", -1)) != TARGET_HORIZON
        or int(
            derived.get("request", {})
            .get("execution", {})
            .get("stop", {})
            .get("maximum_controller_rounds", -1)
        )
        != TARGET_HORIZON
        or source_protocol.get("sha256") != job.get("source_protocol", {}).get("sha256")
        or source_protocol.get("route_contract_sha256")
        != job.get("source_protocol", {}).get("route_contract_sha256")
    ):
        raise ScaffoldContractError(
            f"{job.get('execution_id')} derived protocol binding drifted."
        )
    return json.loads(canonical_json_bytes(scientific))


def predecessor_requirement(
    *, execution_id: str, parent_job: Mapping[str, Any]
) -> dict[str, Any]:
    expected = PENDING_PREDECESSORS[execution_id]
    collision = parent_job.get("collision")
    if not isinstance(collision, Mapping):
        raise ScaffoldContractError(f"{execution_id} lacks its predecessor collision.")
    observed = {
        "cluster_id": int(collision.get("cluster_id", -1)),
        "proc_id": int(collision.get("proc_id", -1)),
        "source_execution_id": str(collision.get("source_execution_id", "")),
    }
    if observed != expected:
        raise ScaffoldContractError(f"{execution_id} predecessor mapping drifted.")
    return digested(
        {
            "schema": PREDECESSOR_REQUIREMENT_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_id": execution_id,
            "source_horizon": SOURCE_HORIZON,
            "target_horizon": TARGET_HORIZON,
            "predecessor": dict(expected),
            "binding_path": (
                f"{EXTERNAL_EVIDENCE_RELATIVE_ROOT}/"
                f"{PREDECESSOR_BINDINGS_DIR}/{execution_id}.json"
            ),
            "resume_archive_path": (
                f"{EXTERNAL_EVIDENCE_RELATIVE_ROOT}/{NEW_RESUME_INPUTS_DIR}/"
                f"{execution_id}.tar.gz"
            ),
            "required_binding_schema": PREDECESSOR_BINDING_SCHEMA,
            "required_checks": [
                "exact_cluster_and_proc_terminal_history",
                "worker_exit_zero_and_round_50_completion",
                "retrieved_attempt_archive_size_and_sha256",
                "source_protocol_and_route_contract_identity",
                "pointer_closed_checkpoint_ledger_resume_triplet",
                "compact_resume_archive_size_and_sha256",
            ],
            "status": "missing_fail_closed",
        }
    )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScaffoldContractError(f"{label} must be a mapping.")
    return value


def _bound_path(
    *, binding: Mapping[str, Any], repo_root: Path, label: str
) -> Path:
    relative = safe_relative_path(binding.get("path"), label=f"{label} path")
    path = repo_root / relative
    verify_exact_binding(path, binding, label=label, rehash=True)
    return path


def _load_bound_controlled_json(
    *, binding: Mapping[str, Any], repo_root: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _bound_path(binding=binding, repo_root=repo_root, label=label)
    try:
        loaded_path, payload, parsed = load_controlled_json_file(
            path, label=label
        )
    except ControlledCycleArchiveError as exc:
        raise ScaffoldContractError(str(exc)) from exc
    if (
        loaded_path != path.absolute()
        or hashlib.sha256(payload).hexdigest() != binding.get("sha256")
        or binding.get("canonical_sha256") != parsed.get("sha256")
    ):
        raise ScaffoldContractError(f"{label} canonical binding drifted.")
    return path, parsed


def _validate_scheduler_terminal_receipt(
    *,
    binding: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    _path, receipt = _load_bound_controlled_json(
        binding=binding,
        repo_root=repo_root,
        label="scheduler terminal receipt",
    )
    if (
        receipt.get("schema") != SCHEDULER_TERMINAL_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("execution_id")
        != predecessor["source_execution_id"]
        or receipt.get("cluster_id") != predecessor["cluster_id"]
        or receipt.get("proc_id") != predecessor["proc_id"]
        or receipt.get("job_status") != 4
        or receipt.get("exit_code") != 0
        or isinstance(receipt.get("num_job_starts"), bool)
        or not isinstance(receipt.get("num_job_starts"), int)
        or int(receipt["num_job_starts"]) < 1
        or not isinstance(receipt.get("completion_epoch"), int)
        or int(receipt["completion_epoch"]) <= 0
        or receipt.get("source") != "condor_history_exact_cluster_proc"
    ):
        raise ScaffoldContractError(
            "Scheduler terminal receipt does not close the exact predecessor."
        )
    return receipt


def _validate_completion_receipt_and_attempt(
    *,
    receipt_binding: Mapping[str, Any],
    attempt_binding: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    scientific_anchor: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _receipt_path, receipt = _load_bound_controlled_json(
        binding=receipt_binding,
        repo_root=repo_root,
        label="controlled-cycle retrieval completion receipt",
    )
    execution = _mapping(receipt.get("execution"), label="receipt execution")
    retrieval = _mapping(receipt.get("retrieval"), label="receipt retrieval")
    local_archive = _mapping(
        retrieval.get("local_archive"), label="receipt local archive"
    )
    archive_validation = _mapping(
        receipt.get("archive_validation"), label="receipt archive validation"
    )
    worker_attempt = _mapping(
        archive_validation.get("worker_attempt_receipt"),
        label="receipt worker attempt",
    )
    release = _mapping(receipt.get("release"), label="receipt release")
    receipt_bindings = _mapping(
        receipt.get("bindings"), label="receipt bindings"
    )
    if (
        receipt.get("schema") != CONTROLLED_COMPLETION_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("completion_classification")
        != "worker_exit_zero_archive_fully_authenticated"
        or execution.get("execution_id")
        != predecessor["source_execution_id"]
        or execution.get("cluster_id") != predecessor["cluster_id"]
        or execution.get("proc_id") != predecessor["proc_id"]
        or not isinstance(execution.get("attempt_ordinal"), int)
        or int(execution["attempt_ordinal"]) < 1
        or local_archive != attempt_binding
        or retrieval.get("remote_archive_sha256")
        != attempt_binding.get("sha256")
        or retrieval.get("remote_archive_size_bytes")
        != attempt_binding.get("size_bytes")
        or retrieval.get("remote_local_hash_size_match") is not True
        or release.get("target")
        != f"{predecessor['cluster_id']}.{predecessor['proc_id']}"
        or release.get("scope") != "exact_cluster_proc_only"
        or release.get("exit_code") != 0
        or worker_attempt.get("worker_exit_status") != 0
        or archive_validation.get("gzip_and_full_tar_scan_passed") is not True
        or archive_validation.get(
            "safe_unique_regular_only_member_closure_passed"
        )
        is not True
        or archive_validation.get(
            "worker_inventory_hash_size_closure_passed"
        )
        is not True
        or archive_validation.get("authority_byte_identity_passed") is not True
    ):
        raise ScaffoldContractError(
            "Controlled-cycle completion receipt relation closure failed."
        )

    job_binding = _mapping(receipt_bindings.get("job"), label="receipt job")
    authorization_binding = _mapping(
        receipt_bindings.get("authorization"), label="receipt authorization"
    )
    activation_binding = _mapping(
        receipt_bindings.get("activation_manifest"),
        label="receipt activation manifest",
    )
    job_path, job = _load_bound_controlled_json(
        binding=job_binding, repo_root=repo_root, label="predecessor job"
    )
    protocol = _mapping(job.get("protocol"), label="predecessor job protocol")
    if (
        job.get("execution_id") != predecessor["source_execution_id"]
        or job.get("horizon") != SOURCE_HORIZON
        or job.get("active_gradient_policy") != ACTIVE_GRADIENT_POLICY
        or job.get("resource_weighting_scope") != RESOURCE_WEIGHTING_SCOPE
        or job.get("phase1_cost_term") != "disabled_for_phase1_only"
        or protocol.get("sha256")
        != scientific_anchor.get("source_protocol_sha256")
        or protocol.get("canonical_sha256")
        != scientific_anchor.get("source_protocol_canonical_sha256")
    ):
        raise ScaffoldContractError(
            "Retrieved predecessor job does not match the scientific anchor."
        )
    attempt_path = _bound_path(
        binding=attempt_binding,
        repo_root=repo_root,
        label="retrieved predecessor archive",
    )
    try:
        validation = validate_attempt_archive(
            attempt_path,
            ExpectedAttempt(
                execution_id=str(predecessor["source_execution_id"]),
                cluster_id=int(predecessor["cluster_id"]),
                proc_id=int(predecessor["proc_id"]),
                job_path=job_path,
                authorization_path=_bound_path(
                    binding=authorization_binding,
                    repo_root=repo_root,
                    label="predecessor execution authorization",
                ),
                activation_manifest_path=_bound_path(
                    binding=activation_binding,
                    repo_root=repo_root,
                    label="predecessor activation manifest",
                ),
                source_archive_sha256=str(
                    receipt_bindings.get("source_archive_sha256", "")
                ),
                image_sha256=str(receipt_bindings.get("image_sha256", "")),
            ),
        )
    except ControlledCycleArchiveError as exc:
        raise ScaffoldContractError(str(exc)) from exc
    if (
        validation.get("status") != "passed"
        or validation.get("archive") != attempt_binding
        or validation.get("worker_attempt_receipt") != worker_attempt
        or validation.get("bindings", {}).get("job") != job_binding
        or validation.get("bindings", {}).get("authorization")
        != authorization_binding
        or validation.get("bindings", {}).get("activation_manifest")
        != activation_binding
    ):
        raise ScaffoldContractError(
            "Controlled-cycle archive revalidation differs from its receipt."
        )
    return receipt, validation


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"active_prefix_checkpoint_count": 0}
    ledger: dict[str, Any] = {}
    resume: dict[str, Any] = {}
    with checkpoint_path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream):
            if (
                prefix == "adapt_vqe.active_prefix_checkpoints.item"
                and event == "start_map"
            ):
                metadata["active_prefix_checkpoint_count"] += 1
            elif prefix == "checkpoint.depth" and event in {"integer", "number"}:
                metadata["checkpoint_depth"] = int(value)
            elif prefix == "adapt_vqe.history_count" and event in {"integer", "number"}:
                metadata["history_count"] = int(value)
            elif (
                prefix == "adapt_vqe.history_checkpoint_complete"
                and event == "boolean"
            ):
                metadata["history_checkpoint_complete"] = value
            elif prefix == "adapt_vqe.strict_replay.passed" and event == "boolean":
                metadata["strict_replay_passed"] = value
            elif (
                prefix == "adapt_vqe.sr_route_profile_contract_sha256"
                and event == "string"
            ):
                metadata["route_contract_sha256"] = value
            elif (
                prefix.startswith("adapt_vqe.estimator_call_ledger_checkpoint.")
                and event in {"boolean", "integer", "number", "string"}
            ):
                ledger[prefix.rsplit(".", 1)[-1]] = value
            elif (
                prefix.startswith("adapt_vqe.verified_singleton_resume_sidecar.")
                and event in {"boolean", "integer", "number", "string"}
            ):
                resume[prefix.rsplit(".", 1)[-1]] = value
    metadata["estimator_call_ledger_checkpoint"] = ledger
    metadata["verified_singleton_resume_sidecar"] = resume
    return metadata


def _safe_tar_member(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ScaffoldContractError(f"Unsafe tar member: {value!r}")
    return value


def validate_resume_input_contents(
    *,
    resume: Mapping[str, Any],
    repo_root: Path,
    expected_route_contract_sha256: str,
    expected_depth: int,
    expected_archive_path: str | None = None,
) -> dict[str, Any]:
    """Fully authenticate one three-member pointer-closed resume archive."""

    archive_binding = _mapping(resume.get("archive"), label="resume archive")
    archive_relative = safe_relative_path(
        archive_binding.get("path"), label="resume archive path"
    )
    if (
        expected_archive_path is not None
        and archive_relative.as_posix() != expected_archive_path
    ):
        raise ScaffoldContractError("Resume archive is not the exact row shard.")
    archive_path = _bound_path(
        binding=archive_binding,
        repo_root=repo_root,
        label="compact resume archive",
    )
    members = resume.get("members")
    if not isinstance(members, list) or len(members) != 3:
        raise ScaffoldContractError("Resume input does not declare three members.")
    by_path: dict[str, Mapping[str, Any]] = {}
    by_role: dict[str, Mapping[str, Any]] = {}
    source_members: set[str] = set()
    for raw in members:
        row = _mapping(raw, label="resume member")
        path = _safe_tar_member(str(row.get("path", "")))
        role = str(row.get("role", ""))
        source_member = _safe_tar_member(str(row.get("source_member", "")))
        if path in by_path or role in by_role or source_member in source_members:
            raise ScaffoldContractError(
                "Resume member path/role/source identity is duplicated."
            )
        by_path[path] = row
        by_role[role] = row
        source_members.add(source_member)
    required_roles = {
        "checkpoint",
        "estimator_ledger_checkpoint",
        "verified_resume_sidecar",
    }
    if (
        set(by_role) != required_roles
        or resume.get("member_count") != 3
        or resume.get("pointer_closed") is not True
        or resume.get("superseded_sidecars_retained", False) is not False
        or resume.get("checkpoint_path") != by_role["checkpoint"].get("path")
        or resume.get("checkpoint_sha256")
        != by_role["checkpoint"].get("sha256")
    ):
        raise ScaffoldContractError("Resume triplet declaration is not closed.")

    observed: set[str] = set()
    checkpoint_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="ra-r70-checkpoint-", suffix=".json", delete=False
        ) as temporary:
            checkpoint_tmp = Path(temporary.name)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                name = _safe_tar_member(member.name)
                expected = by_path.get(name)
                if (
                    expected is None
                    or name in observed
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != int(expected.get("size_bytes", -1))
                ):
                    raise ScaffoldContractError(
                        f"Compact resume contains an unsafe/unexpected member: {name}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ScaffoldContractError(f"Unreadable resume member: {name}")
                digest = hashlib.sha256()
                size = 0
                output = (
                    checkpoint_tmp.open("wb")
                    if expected.get("role") == "checkpoint"
                    else None
                )
                try:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        if output is not None:
                            output.write(block)
                finally:
                    if output is not None:
                        output.close()
                if (
                    size != member.size
                    or digest.hexdigest() != expected.get("sha256")
                ):
                    raise ScaffoldContractError(
                        f"Compact resume member hash/size drifted: {name}"
                    )
                observed.add(name)
        if observed != set(by_path):
            raise ScaffoldContractError("Compact resume member closure is incomplete.")
        assert checkpoint_tmp is not None
        metadata = _checkpoint_metadata(checkpoint_tmp)
    except (OSError, EOFError, tarfile.TarError, ijson.JSONError) as exc:
        raise ScaffoldContractError("Compact resume archive/checkpoint is invalid.") from exc
    finally:
        if checkpoint_tmp is not None:
            checkpoint_tmp.unlink(missing_ok=True)

    ledger = _mapping(
        metadata.get("estimator_call_ledger_checkpoint"),
        label="checkpoint ledger pointer",
    )
    sidecar = _mapping(
        metadata.get("verified_singleton_resume_sidecar"),
        label="checkpoint resume pointer",
    )
    for role, pointer in (
        ("estimator_ledger_checkpoint", ledger),
        ("verified_resume_sidecar", sidecar),
    ):
        member = by_role[role]
        source_member = str(member.get("source_member", ""))
        if (
            pointer.get("status") != "complete"
            or pointer.get("sha256") != member.get("sha256")
            or (
                pointer.get("size_bytes") is not None
                and int(pointer["size_bytes"])
                != int(member.get("size_bytes", -1))
            )
            or PurePosixPath(str(pointer.get("path", ""))).name
            != PurePosixPath(source_member).name
            or PurePosixPath(str(member.get("path", ""))).name
            != PurePosixPath(source_member).name
            or (role == "verified_resume_sidecar" and pointer.get("enabled") is not True)
        ):
            raise ScaffoldContractError(
                f"Checkpoint {role} pointer does not bind its tar member."
            )
    if (
        metadata.get("checkpoint_depth") != expected_depth
        or metadata.get("history_count") != expected_depth
        or metadata.get("active_prefix_checkpoint_count") != expected_depth
        or metadata.get("history_checkpoint_complete") is not True
        or metadata.get("strict_replay_passed") is not True
        or metadata.get("route_contract_sha256")
        != expected_route_contract_sha256
    ):
        raise ScaffoldContractError(
            f"Compact checkpoint is not an authenticated round-{expected_depth} prefix."
        )

    return {
        "archive_path": archive_path,
        "metadata": metadata,
        "members_by_role": by_role,
    }


def _validate_compact_resume_archive(
    *,
    resume: Mapping[str, Any],
    requirement: Mapping[str, Any],
    attempt_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    scientific_anchor = _mapping(
        requirement.get("scientific_anchor"), label="scientific anchor"
    )
    validated = validate_resume_input_contents(
        resume=resume,
        repo_root=repo_root,
        expected_route_contract_sha256=str(
            scientific_anchor.get("route_contract_sha256", "")
        ),
        expected_depth=SOURCE_HORIZON,
        expected_archive_path=str(requirement.get("resume_archive_path", "")),
    )
    by_role = _mapping(
        validated.get("members_by_role"), label="validated resume members"
    )

    expected_attempt_members = {
        str(row.get("source_member")): row for row in by_role.values()
    }
    if len(expected_attempt_members) != 3 or any(
        not name for name in expected_attempt_members
    ):
        raise ScaffoldContractError("Resume source-member closure is incomplete.")
    predecessor = _mapping(requirement.get("predecessor"), label="predecessor")
    checkpoint_source = str(by_role["checkpoint"].get("source_member", ""))
    if (
        predecessor["source_execution_id"] not in checkpoint_source
        or not checkpoint_source.endswith("/checkpoints/current.json")
    ):
        raise ScaffoldContractError("Checkpoint source member identity drifted.")
    found: set[str] = set()
    try:
        with tarfile.open(attempt_path, "r:gz") as attempt:
            for member in attempt:
                expected = expected_attempt_members.get(member.name)
                if expected is None:
                    continue
                if (
                    member.name in found
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or member.size != int(expected.get("size_bytes", -1))
                ):
                    raise ScaffoldContractError(
                        f"Attempt resume source member is unsafe: {member.name}"
                    )
                stream = attempt.extractfile(member)
                if stream is None:
                    raise ScaffoldContractError(
                        f"Attempt resume source member is unreadable: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                if (
                    size != member.size
                    or digest.hexdigest() != expected.get("sha256")
                ):
                    raise ScaffoldContractError(
                        f"Compact member does not match attempt source: {member.name}"
                    )
                found.add(member.name)
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ScaffoldContractError("Attempt resume-member scan failed.") from exc
    if found != set(expected_attempt_members):
        raise ScaffoldContractError("Attempt lacks a compact-resume source member.")
    return dict(_mapping(validated.get("metadata"), label="resume metadata"))


def validate_predecessor_binding(
    *,
    binding: Mapping[str, Any],
    requirement: Mapping[str, Any],
    repo_root: Path,
    rehash_resume: bool,
) -> dict[str, Any]:
    del rehash_resume  # Security-critical archives are always fully rehashed.
    verify_self_digest(binding, label="predecessor binding")
    predecessor = _mapping(requirement.get("predecessor"), label="predecessor")
    resume = _mapping(binding.get("resume_input"), label="resume input")
    attempt = _mapping(binding.get("attempt_archive"), label="attempt archive")
    completion = _mapping(
        binding.get("retrieval_completion_receipt"),
        label="retrieval completion receipt",
    )
    scheduler_receipt = _mapping(
        binding.get("scheduler_terminal_receipt"),
        label="scheduler terminal receipt",
    )
    expected_scientific = _mapping(
        requirement.get("scientific_anchor"), label="scientific anchor"
    )
    if (
        binding.get("schema") != PREDECESSOR_BINDING_SCHEMA
        or binding.get("package_id") != PACKAGE_ID
        or binding.get("execution_id") != requirement.get("execution_id")
        or binding.get("source_horizon") != SOURCE_HORIZON
        or binding.get("target_horizon") != TARGET_HORIZON
        or binding.get("status") != "passed"
        or binding.get("scientific_anchor") != expected_scientific
    ):
        raise ScaffoldContractError("Predecessor binding header drifted.")
    _validate_scheduler_terminal_receipt(
        binding=scheduler_receipt,
        predecessor=predecessor,
        repo_root=repo_root,
    )
    _receipt, _validation = _validate_completion_receipt_and_attempt(
        receipt_binding=completion,
        attempt_binding=attempt,
        predecessor=predecessor,
        scientific_anchor=expected_scientific,
        repo_root=repo_root,
    )
    attempt_path = _bound_path(
        binding=attempt,
        repo_root=repo_root,
        label="retrieved predecessor archive",
    )
    _validate_compact_resume_archive(
        resume=resume,
        requirement=requirement,
        attempt_path=attempt_path,
        repo_root=repo_root,
    )
    return json.loads(canonical_json_bytes(binding))


def transfer_path_is_regular_file(path: str) -> bool:
    candidate = PurePosixPath(path)
    if candidate.name in {"resume_inputs", NEW_RESUME_INPUTS_DIR}:
        return False
    return candidate.suffix in {".json", ".gz", ".sif", ".sh"} or candidate.name.endswith(
        ".tar.gz"
    )
