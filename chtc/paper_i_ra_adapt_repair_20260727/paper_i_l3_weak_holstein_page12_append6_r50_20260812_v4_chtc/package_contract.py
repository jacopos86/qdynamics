#!/usr/bin/env python3
"""Closed contract for the matched six-cell L=3 weak-Holstein package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

PACKAGE_ID = "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4_chtc"
CAMPAIGN_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v4"
BUNDLE_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v4"
BATCH_NAME = "paper-i-l3-weak-holstein-page12-append6-r50-20260812-v4"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"
TARGET_HORIZON = 50
REGIMES = ("weak_weak", "intermediate_weak", "strong_weak_u8")
METHODS = ("ra_page12", "append_adapt")
RA_ALGORITHM_ID = (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase1_phase2_phase3_"
    "qiskit_phase2_phase3_plateau_no_lanes_v1"
)
APPEND_ALGORITHM_ID = "paper_i_append_adapt_v1"
RA_ROUTE_ID = "ra_l3_page12_plateau"
APPEND_ROUTE_ID = "append_l3_conventional_unwhitened"
TARGET_ROUTE_CONTRACT_SHA256 = (
    "8d5f9a53d79c30abba5c26b9bba68751dea3122b2f692021a44e7db260748e83"
)
APPLICATION_SOURCE_LOCK_KEY = "paper_i_l3_page12_application_source_sha256"
REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
ACTIVE_GRADIENT_RA = "stationary_source_response_v1"
ACTIVE_GRADIENT_APPEND = "measured_residual_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_l3_page12_global_singleton_gradient_phase0_candidate_adapter_v1"
)
APPEND_RUNTIME_SOURCE_DEPENDENCIES = (
    {
        "path": "pipelines/exact_bench/generic_static_adapt_variants.py",
        "sha256": (
            "1a82945bfcc8e4273c09e2c4f24fb7c1f85df71bb1b952163afe8f349d4262e1"
        ),
        "size_bytes": 490_408,
    },
)
REQUIRED_ROUTE_SOURCE_PATHS = (
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/adapters.py",
    "pipelines/static_adapt/ra_adapt/append.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/l3_page12.py",
    "pipelines/static_adapt/ra_adapt/phase0.py",
    "pipelines/static_adapt/ra_adapt/pools.py",
    "pipelines/static_adapt/sr_snake/_selection.py",
    "pipelines/exact_bench/generic_static_adapt_variants.py",
)
RESOURCE_ENVELOPES = {
    "ra_page12": {
        "request_cpus": 4,
        "request_memory_mb": 65_536,
        "request_disk_mb": 81_920,
        "max_runtime_seconds": 259_200,
        "basis": "v3_resource_only_rightsizing_ra_page12_v1",
    },
    "append_adapt": {
        "request_cpus": 1,
        "request_memory_mb": 49_152,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "v3_resource_only_rightsizing_conventional_append_v1",
    },
}
STAGING_OUTPUT_ROOT = (
    "/staging/jsstrobel/paper_i_l3_weak_holstein_page12_append6_r50_20260812_v4"
)

V3_PACKAGE_RELATIVE_ROOT = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc"
)
V3_PACKAGE_ID = (
    "paper_i_l3_weak_holstein_page12_append6_r50_20260812_v3_chtc"
)
V3_CAMPAIGN_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v3"
V3_BUNDLE_ID = "paper_i_l3_weak_holstein_page12_append6_r50_v3"
V3_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "da24fd9467318dcf5786883104945decee3b57d460f3097cd7995ccc64268edf"
)
V3_PACKAGE_MANIFEST_FILE_SHA256 = (
    "bf7416fc0601a812ffc58a6345cf5eaa895f95d8d7ef3837f462d842860ee320"
)
V3_PACKAGE_MANIFEST_SIZE_BYTES = 11_485
V3_SOURCE_ARCHIVE_SHA256 = (
    "2aa61620dee19e9dcadb9e90a1008969e8c1ce752f1ad7ee9ccfdc94c7973400"
)
V3_SOURCE_ARCHIVE_SIZE_BYTES = 1_876_644
V3_IMPLEMENTATION_SOURCE_INVENTORY_SHA256 = (
    "9b7e6da4b64637f7a5a6873040016f3cff82e8a635e5808a400559bf1114bb09"
)
SCIENTIFIC_EQUIVALENCE_RELATIVE = "v3_scientific_equivalence.json"

PACKAGE_MANIFEST_SCHEMA = "paper_i_l3_weak_holstein_matched_package_manifest_v1"
JOB_SCHEMA = "paper_i_l3_weak_holstein_matched_job_v1"
AUTHORIZATION_SCHEMA = "paper_i_l3_weak_holstein_execution_authorization_v1"
ACTIVATION_REQUEST_SCHEMA = "paper_i_l3_weak_holstein_activation_request_v1"
ACTIVATION_MANIFEST_SCHEMA = "paper_i_l3_weak_holstein_activation_manifest_v1"
EXECUTION_PLAN_SCHEMA = "paper_i_l3_weak_holstein_execution_plan_v1"
SOURCE_ARCHIVE_MANIFEST_SCHEMA = "paper_i_l3_weak_holstein_source_archive_manifest_v1"

CONTROL_FILES = (
    "package_contract.py", "build_package.py", "activate_package.py",
    "run_cell.py", "validate_package.py", "probe_image_runtime.py",
    "execute_authorized_job.sh", "submit.sub.in",
)
GENERATED_PATHS = (
    "source_authority", "bundle_materialization", "source", "jobs",
    "queue.tsv", "execution_plan.json", "source_lock_audit.json",
    SCIENTIFIC_EQUIVALENCE_RELATIVE, "package_manifest.json",
)

class PackageContractError(RuntimeError):
    """Fail-closed package or worker-contract violation."""

def execution_id(regime_id: str, method: str) -> str:
    if regime_id not in REGIMES or method not in METHODS:
        raise PackageContractError("Unknown matched L3 execution coordinate.")
    suffix = (
        "ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_plateau"
        if method == "ra_page12" else "append_conventional_unwhitened"
    )
    return f"l3_weak_holstein__{regime_id}__nph3__{suffix}"

def source_lock_id(regime_id: str) -> str:
    if regime_id not in REGIMES:
        raise PackageContractError("Unknown weak-Holstein regime.")
    return f"l3_weak_holstein__{regime_id}__nph3__application_v1"

def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(regime, method) for regime in REGIMES for method in METHODS)

def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value); payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload

def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
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
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict): raise PackageContractError(f"{label} must be an object.")
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
    try: display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc: raise PackageContractError(f"Binding escaped package: {path}") from exc
    result = {"path": display, "sha256": sha256_file(resolved),
              "size_bytes": resolved.stat().st_size}
    if canonical:
        payload = load_json(resolved, label=display)
        result["canonical_sha256"] = verify_self_digest(payload, label=display)
    return result

def repo_root_from_script(script: str | Path) -> Path:
    current = Path(script).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (candidate / "pipelines/static_adapt").is_dir():
            return candidate
    raise PackageContractError("Could not resolve active repository root.")
