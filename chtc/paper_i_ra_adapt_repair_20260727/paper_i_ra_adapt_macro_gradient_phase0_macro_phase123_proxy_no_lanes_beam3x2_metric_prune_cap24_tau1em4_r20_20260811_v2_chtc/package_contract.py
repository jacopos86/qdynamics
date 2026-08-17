#!/usr/bin/env python3
"""Closed contract for the macro Phase-0 beam/metric-pruning ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "beam3x2_metric_prune_cap24_tau1em4_r20_20260811_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "beam3x2_metric_prune_cap24_tau1em4_r20_v2"
)
BUNDLE_ID = (
    "ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_beam3x2_"
    "metric_prune_cap24_tau1em4_r20_v2"
)
BATCH_NAME = "paper-i-page13-macro-beam-metric-r20-20260811-v2"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
ALGORITHM_ID = (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase1_phase2_phase3_"
    "proxy_plateau_no_lanes_v1"
)
ROUTE_ID = (
    "ra_macro_gradient_phase0_macro_phase123_proxy_no_lanes_plateau_"
    "beam3x2_metric_prune"
)
INHERITED_SOURCE_LOCK_ROUTE_ID = "ra_macro_plateau"
SOURCE_ROUTE_PROFILE = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_no_prune_"
    "macro_only_physical_lanes_insertion_commutation_plateau_v2"
)
SOURCE_ROUTE_CONTRACT_SHA256 = (
    "e7b17287fb21adf703101f44da31cdf4e716d0752600aa36dd30691384d8fbd7"
)
TARGET_PARENT_ROUTE_PROFILE = (
    f"{SOURCE_ROUTE_PROFILE}__pruning-metric__beam-fork_local"
)
TARGET_ROUTE_SUFFIX = (
    "macro_abs_gradient_phase0_then_macro_phase1_then_identity_macro_"
    "phase2_phase3_proxy_no_lanes_v1"
)
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__macro_generator_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__pruning-metric__beam-fork_local__"
    f"{TARGET_ROUTE_SUFFIX}"
)
TARGET_PARENT_ROUTE_CONTRACT_SHA256 = (
    "1cebfef5b79ed86fc40072f896f6921da202c004e09025750e86e130141154eb"
)
TARGET_ROUTE_CONTRACT_SHA256 = (
    "93e53e05fbcdcf23bf589c88374e0181d18e5d1abd99c68be3c80c6c37f1a9a0"
)
PRUNING_POLICY = "metric"
BEAM_LIVE_BRANCHES = 3
BEAM_CHILDREN_PER_PARENT = 2
BEAM_MAXIMUM_CHILDREN_PER_ROUND = 6
BEAM_S_ALG_WEIGHT = 0.005
STRUCTURAL_PROXY_MODE = "marrakesh_graph_span_v1"
STRUCTURAL_PROXY_COST_SOURCE = (
    "marrakesh_graph_span_structural_proxy_v1"
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
EXECUTION_ID_PREFIX = "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric"
STAGE_ID = "macro_gradient_phase0_proxy_no_lanes_beam3x2_metric_candidate"

WEAK_HORIZON = 20
STRONG_HORIZON = 20
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
    "cap24_tau1em4_r50_20260810_v1_chtc"
)
BASELINE_BUNDLE_ID = (
    "ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_cap24_"
    "tau1em4_r50_v1"
)
BASELINE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "4345ee2c21f84d55cfdf3cb18a6bae8ad957899e708fc6d96e1f5a6e020c0f97"
)
BASELINE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "8559b91825bb7659a1a3591ac5fcbdb1d57eaa6496ae31df6da44c227e62dbac"
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
    "weak_weak": {
        "request_cpus": 4,
        "request_memory_mb": 40_960,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_weak_weak_nph3_r20_v1",
    },
    "intermediate_weak": {
        "request_cpus": 4,
        "request_memory_mb": 40_960,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_intermediate_weak_nph3_r20_v1",
    },
    "strong_weak_u8": {
        "request_cpus": 4,
        "request_memory_mb": 49_152,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_strong_weak_u8_nph3_r20_v1",
    },
    "weak_strong": {
        "request_cpus": 4,
        "request_memory_mb": 57_344,
        "request_disk_mb": 81_920,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_weak_strong_nph7_r20_v1",
    },
    "intermediate_strong": {
        "request_cpus": 4,
        "request_memory_mb": 57_344,
        "request_disk_mb": 81_920,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_intermediate_strong_nph7_r20_v1",
    },
    "strong_strong_u8": {
        "request_cpus": 4,
        "request_memory_mb": 65_536,
        "request_disk_mb": 81_920,
        "max_runtime_seconds": 259_200,
        "basis": "page13_macro_beam_metric_strong_strong_u8_nph7_r20_v1",
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
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_package_manifest_v2"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_job_v2"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
    "execution_authorization_v2"
)
ACTIVATION_REQUEST_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
    "activation_request_v2"
)
ACTIVATION_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
    "activation_manifest_v2"
)
EXECUTION_PLAN_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
    "execution_plan_v2"
)
SOURCE_AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_"
    "source_authority_v2"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_source_lock_audit_v2"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_macro_gradient_phase0_proxy_no_lanes_source_archive_"
    "manifest_v2"
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
