#!/usr/bin/env python3
"""Closed contract for the named L=3 intermediate-weak Page-12 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v2_chtc"
CAMPAIGN_ID = "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_v2"
BUNDLE_ID = "ra_adapt_l3_intermediate_weak_page12_r50_v2"
BATCH_NAME = "paper-i-l3-intermediate-weak-page12-r50-20260811-v2"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_"
    "phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
SOURCE_ALGORITHM_ID = "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
ROUTE_ID = "ra_l3_intermediate_weak_page12_plateau"
SOURCE_ROUTE_PROFILE = (
    "supported_projected_generalized_source_metric_no_overlap_trust_"
    "full_response_symmetric_cost_no_prune_insertion_commutation_plateau_v2"
)
SOURCE_ROUTE_CONTRACT_SHA256 = (
    "aa669d7f0c3621d9ddf7f8595f96333c56b536c8fc79547607e76d8d91d4b6ff"
)
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__global_singleton_abs_gradient_phase0_"
    "then_singleton_phase1_then_qiskit_phase2_phase3_no_lanes_v1"
)
TARGET_ROUTE_CONTRACT_SHA256 = (
    "8d5f9a53d79c30abba5c26b9bba68751dea3122b2f692021a44e7db260748e83"
)
APPLICATION_SOURCE_LOCK_KEY = "paper_i_l3_page12_application_source_sha256"
APPLICATION_SOURCE_SHA256 = (
    "7ef4bdc24f4dbd751bdfeebed3ab26be1dfece0a33331ba18eff38b35cfad70c"
)
SAME_CUTOFF_REFERENCE_RECEIPT_SHA256 = (
    "079ef700ed8fd478ccd45b64df740815c2a68ec10dc280bd7ec84bcf71dddd04"
)
BACKEND_COMPILE_SCOPE = "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
SELECTOR_COMPILE_COST_PHASE_REUSE = (
    "phase_ii_phase_iii_shared_oracle_snapshot_and_cache_v1"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_l3_page12_global_singleton_gradient_phase0_candidate_adapter_v1"
)
PHASE0_VARIANT = "global_singleton"
PHASE0_POLICY = "global_singleton_absolute_gradient_shortlist_v1"
PHASE0_SHORTLIST_SIZE = 24
EXPECTED_CANDIDATE_FUNNEL = (
    "global_singleton_gradient_phase0_shortlist_then_singleton_phase1_"
    "shortlist_then_singleton_phase2_then_singleton_phase3_v1"
)
EXECUTION_ID = (
    "l3_page12__intermediate_weak__nph1__"
    "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
)
SOURCE_LOCK_ID = "l3_page12__intermediate_weak__nph1__application_v1"
STAGE_ID = "l3_page12_intermediate_weak_candidate"
TARGET_HORIZON = 50
RESOURCE_ENVELOPE = {
    "request_cpus": 4,
    "request_memory_mb": 49_152,
    "request_disk_mb": 61_440,
    "max_runtime_seconds": 259_200,
    "basis": "explicit_user_l3_page12_r50_envelope_v1",
}
STAGING_OUTPUT_ROOT = (
    "/staging/jsstrobel/paper_i_ra_adapt_l3_intermediate_weak_page12_"
    "r50_20260811_v2"
)

REQUIRED_PHASE3_QISKIT_SOURCE_PATHS = (
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/adapters.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/l3_page12.py",
    "pipelines/static_adapt/ra_adapt/phase0.py",
    "pipelines/static_adapt/ra_adapt/pools.py",
    "pipelines/static_adapt/sr_snake/_selection.py",
)

PACKAGE_MANIFEST_SCHEMA = "paper_i_ra_adapt_l3_page12_package_manifest_v1"
JOB_SCHEMA = "paper_i_ra_adapt_l3_page12_job_v1"
AUTHORIZATION_SCHEMA = "paper_i_ra_adapt_l3_page12_execution_authorization_v1"
ACTIVATION_REQUEST_SCHEMA = "paper_i_ra_adapt_l3_page12_activation_request_v1"
ACTIVATION_MANIFEST_SCHEMA = "paper_i_ra_adapt_l3_page12_activation_manifest_v1"
EXECUTION_PLAN_SCHEMA = "paper_i_ra_adapt_l3_page12_execution_plan_v1"
SOURCE_ARCHIVE_MANIFEST_SCHEMA = "paper_i_ra_adapt_l3_page12_source_archive_manifest_v1"

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "activate_package.py",
    "run_cell.py",
    "validate_package.py",
    "probe_image_runtime.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
)
GENERATED_PATHS = (
    "source_authority",
    "bundle_materialization",
    "source",
    "jobs",
    "queue.tsv",
    "execution_plan.json",
    "source_lock_audit.json",
    "package_manifest.json",
)


class PackageContractError(RuntimeError):
    """Fail-closed package or worker-contract violation."""


def source_lock_id(_regime_id: str = "intermediate_weak", _nph: int = 1) -> str:
    return SOURCE_LOCK_ID


def execution_id(_regime_id: str = "intermediate_weak", _nph: int = 1) -> str:
    return EXECUTION_ID


def expected_execution_ids() -> tuple[str, ...]:
    return (EXECUTION_ID,)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    observed = canonical_sha256(unsigned)
    if value.get("sha256") != observed:
        raise PackageContractError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return Path(*pure.parts)


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageContractError(f"Missing or unsafe binding target: {path}")
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(f"Binding target escaped package: {path}") from exc
    result: dict[str, Any] = {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if canonical:
        payload = load_json(resolved, label=display)
        result["canonical_sha256"] = verify_self_digest(payload, label=display)
    return result


def repo_root_from_script(script: str | Path) -> Path:
    current = Path(script).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines/static_adapt"
        ).is_dir():
            return candidate
    raise PackageContractError("Could not resolve the active repository root.")
