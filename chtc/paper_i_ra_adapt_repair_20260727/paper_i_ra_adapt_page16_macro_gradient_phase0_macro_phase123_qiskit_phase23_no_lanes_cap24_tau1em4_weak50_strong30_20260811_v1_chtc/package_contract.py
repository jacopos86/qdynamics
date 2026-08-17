#!/usr/bin/env python3
"""Closed contract for the Page-16 macro Phase-II/III Qiskit run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_weak50_strong30_20260811_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_"
    "phase23_no_lanes_cap24_tau1em4_weak50_strong30_v1"
)
BUNDLE_ID = (
    "ra_adapt_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_cap24_tau1em4_weak50_strong30_v1"
)
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
ROUTE_ID = (
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_plateau"
)
INHERITED_SOURCE_LOCK_ROUTE_ID = "ra_macro_plateau"
SOURCE_ROUTE_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_"
    "macro_only_physical_lanes_insertion_commutation_plateau_v2"
)
SOURCE_ROUTE_CONTRACT_SHA256 = (
    "e7b17287fb21adf703101f44da31cdf4e716d0752600aa36dd30691384d8fbd7"
)
TARGET_ROUTE_SUFFIX = (
    "macro_abs_gradient_phase0_then_macro_phase1_then_identity_macro_"
    "phase2_phase3_qiskit_no_lanes_v1"
)
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__macro_generator_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    f"{TARGET_ROUTE_SUFFIX}"
)
STRUCTURAL_PROXY_MODE = "marrakesh_graph_span_v1"
BACKEND_COMPILE_SCOPE = (
    "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
)
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
SELECTOR_COMPILE_COST_PHASE_REUSE = (
    "phase_ii_phase_iii_shared_oracle_snapshot_and_cache_v1"
)
SOURCE_ACTIVATION_POLICY = (
    "chdir_source_then_purge_ambient_modules_and_paths_before_sealed_import_v2"
)
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "macro_generator_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_candidate_adapter_v1"
)
PHASE0_POLICY = "standard_adapt_abs_gradient_macro_phase0_v1"
PHASE0_SHORTLIST_SIZE = 24
EXPECTED_CANDIDATE_FUNNEL = (
    "macro_gradient_phase0_shortlist_then_macro_phase1_then_identity_macro_"
    "phase2_then_macro_phase3_v1"
)
EXECUTION_ID_PREFIX = "page16_macro_gradient_phase0_phase23_qiskit_no_lanes"
STAGE_ID = "page16_macro_gradient_phase0_phase23_qiskit_candidate"

WEAK_HORIZON = 50
STRONG_HORIZON = 30
REGIME_ROWS: tuple[tuple[str, int, int], ...] = (
    ("weak_weak", 3, WEAK_HORIZON),
    ("intermediate_weak", 3, WEAK_HORIZON),
    ("strong_weak_u8", 3, WEAK_HORIZON),
    ("weak_strong", 7, STRONG_HORIZON),
    ("intermediate_strong", 7, STRONG_HORIZON),
    ("strong_strong_u8", 7, STRONG_HORIZON),
)

BASELINE_PACKAGE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "cap24_tau1em4_r50_20260810_v3_chtc"
)
BASELINE_BUNDLE_ID = (
    "ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_cap24_"
    "tau1em4_r50_v3"
)
BASELINE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "b2a3a1b1f9fab009c8a4750b05cb533528a5cb5abefbe533bf3fd9aaf12574a0"
)
BASELINE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "1ec606f6162a6a5c83b8f618112cb36ed271d2d233543c8c70d81f5098e3f7fb"
)
BASELINE_ROUTE_CONTRACT_SHA256 = (
    "1b2f7254a96a27a7f2a262f1b4bc19c886b421a9cbaa5e24c95e354a02f2cf45"
)
BASELINE_ROUTE_PROFILE = (
    "paper_i_ra_adapt__macro_generator_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__"
    "macro_abs_gradient_phase0_then_macro_phase1_then_identity_macro_"
    "phase2_phase3_proxy_no_lanes_v1"
)

SOURCE_MATERIALIZATION = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13"
)
SOURCE_BUNDLE_ID = "ra_repair_stationary_late_core_v1"
SOURCE_FINAL_RECEIPT_FILE_SHA256 = (
    "d9219d94db6f75d65842b828642e833d3c17eef0534516d0ef6dc2de08f6b415"
)
SOURCE_FINAL_RECEIPT_CANONICAL_SHA256 = (
    "60f7c5cd29fe0c7c9f62c6dc8a8de2581eaad9a322ebfe0fab5d8c6220576274"
)
SOURCE_LOCKS_FILE_SHA256 = (
    "4b65d21bde548345a4dc47995974271f110c23bb6ead2237b1bccea18e0ae8cd"
)
SOURCE_LOCKS_CANONICAL_SHA256 = (
    "0a5e58daa29f133d3e3aa1406672053cb8e80f6182baed9f89690b4aeec16c17"
)

RESOURCE_ENVELOPES = {
    3: {
        "request_cpus": 4,
        "request_memory_mb": 32_768,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": (
            "page16_macro_phase0_phase23_qiskit_no_lanes_nph3_r50_"
            "page13_v3_matchmaking_envelope_v1"
        ),
    },
    7: {
        "request_cpus": 4,
        "request_memory_mb": 49_152,
        "request_disk_mb": 81_920,
        "max_runtime_seconds": 259_200,
        "basis": (
            "page16_macro_phase0_phase23_qiskit_no_lanes_nph7_r30_"
            "page13_v3_matchmaking_envelope_v1"
        ),
    },
}
REQUIRED_ROUTE_SOURCE_PATHS = (
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/adapters.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/phase0.py",
    "pipelines/static_adapt/ra_adapt/pools.py",
)

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_package_manifest_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_authorization_v1"
)
ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_activation_request_v1"
)
ACTIVATION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_activation_manifest_v1"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_execution_plan_v1"
)
SOURCE_AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_source_authority_v1"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_source_lock_audit_v1"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_page16_macro_phase23_qiskit_source_archive_manifest_v1"
)

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


def source_lock_id(regime_id: str, nph: int) -> str:
    return (
        f"{regime_id}__nph{int(nph)}__"
        f"{INHERITED_SOURCE_LOCK_ROUTE_ID}"
    )


def execution_id(regime_id: str, nph: int) -> str:
    return (
        f"{EXECUTION_ID_PREFIX}__"
        f"{regime_id}__nph{int(nph)}__{ROUTE_ID}"
    )


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(regime, nph) for regime, nph, _ in REGIME_ROWS)


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
