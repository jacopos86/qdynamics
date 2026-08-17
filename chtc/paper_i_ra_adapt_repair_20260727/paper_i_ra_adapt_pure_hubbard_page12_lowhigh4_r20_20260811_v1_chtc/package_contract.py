#!/usr/bin/env python3
"""Closed contract for the four-cell pure-Hubbard Page-12 noise prefix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_"
    "20260811_v1_chtc"
)
CAMPAIGN_ID = "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_v1"
BUNDLE_ID = "ra_adapt_pure_hubbard_page12_lowhigh4_r20_v1"
BATCH_NAME = "paper-i-pure-hubbard-page12-lowhigh4-r20-20260811-v1"
RUN_CLASS = "candidate"
EXECUTION_TARGET = "chtc"

SOURCE_PACKAGE_ID = (
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_"
    "20260811_v3_chtc"
)
SOURCE_PACKAGE_RELATIVE_PATH = (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3_chtc"
)
SOURCE_PACKAGE_MANIFEST_SHA256 = (
    "ed280fa195757fe9b3363b36dc85fdab851d9411a7b5e77f356242fe3944f38b"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "19a453dace0059d3034e0596775c4b10969e73cb85da46c8ddf72fcea910ec5d"
)
SOURCE_HORIZON = 50
SOURCE_REQUEST_MEMORY_MB = 8_192

ALGORITHM_ID = (
    "paper_i_ra_adapt_pure_hubbard_full_noise_global_singleton_gradient_"
    "phase0_phase1_phase2_phase3_qiskit_phase2_phase3_plateau_no_lanes_v1"
)
APPLICATION_SOURCE_LOCK_KEY = (
    "paper_i_pure_hubbard_noise_page12_application_source_sha256"
)
ROUTE_ID = "ra_pure_hubbard_page12_full_noise_plateau"
TARGET_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v1__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1__global_singleton_abs_gradient_phase0_"
    "then_singleton_phase1_then_qiskit_phase2_phase3_no_lanes_v1"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_pure_hubbard_noise_page12_global_singleton_gradient_phase0_"
    "candidate_adapter_v1"
)
TARGET_HORIZON = 20
PHASE0_VARIANT = "global_singleton"
PHASE0_POLICY = "global_singleton_absolute_gradient_shortlist_v1"
PHASE0_SHORTLIST_SIZE = 24
BACKEND_COMPILE_SCOPE = "phase_i_proxy_phase_ii_phase_iii_qiskit_transpile_v1"
SELECTOR_COMPILE_COST_POLICY = (
    "qiskit_full_trial_ansatz_signed_marginal_phase2_phase3_v1"
)
SELECTOR_COMPILE_COST_PHASE_REUSE = (
    "phase_ii_phase_iii_shared_oracle_snapshot_and_cache_v1"
)
EXPECTED_CANDIDATE_FUNNEL = (
    "global_singleton_gradient_phase0_shortlist_then_singleton_phase1_"
    "shortlist_then_singleton_phase2_then_singleton_phase3_v1"
)
INSERTION_POLICY = "cumulative_relative_plateau_commutation_reduced_tau1em4_v1"
PLATEAU_THRESHOLD = 1.0e-4
OPTIMIZER = "powell"
OPTIMIZER_MAXITER = 200
ALGORITHM_SEED = 7
VALUE_NOISE_SEED = 702688422
COHERENT_NOISE_SEED = 20260609
NOISE_TUPLE_ORDER = ("sigma_E", "p1", "p2", "epsilon1", "epsilon2")
NOISE_LEVELS: tuple[tuple[str, tuple[float, float, float, float, float]], ...] = (
    ("low", (1.0e-6, 1.0e-8, 1.0e-7, 2.0e-4, 6.0e-4)),
    ("high", (7.071067811865475e-5, 1.0e-6, 1.0e-5, 2.0e-3, 6.0e-3)),
)
U_VALUES = (1.5, 8.0)
CELL_ROWS: tuple[tuple[float, str, tuple[float, ...]], ...] = tuple(
    (u_value, noise_level, noise_tuple)
    for u_value in U_VALUES
    for noise_level, noise_tuple in NOISE_LEVELS
)
CELL_COUNT = len(CELL_ROWS)
INERT_PACKAGE_STATUS = "passed_inert_four_cells"
ACTIVATION_SCOPE = "prepare_four_cell_chtc_execution_and_submission_v1"

EXECUTION_ID_PREFIX = "pure_hubbard_page12_fullnoise"
STAGE_ID = "pure_hubbard_page12_fullnoise_candidate"

REMOTE_IMAGE_PATH = "chtc/phase3_optuna/image.sif"
REMOTE_IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
STAGING_OUTPUT_ROOT = (
    "/staging/jsstrobel/paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_"
    "r20_20260811_v1"
)

# The source package's four low/high cells exceeded an 8-GiB cgroup limit,
# with scheduler-observed MemoryUsage reaching 9,766 MiB.  Use one common
# 16-GiB operational envelope for the source-identical 20-round recovery.
RESOURCE_ENVELOPE = {
    "request_cpus": 2,
    "request_memory_mb": 16_384,
    "request_disk_mb": 12_288,
    "max_runtime_seconds": 259_200,
    "basis": (
        "pure_hubbard_l2_page12_lowhigh_r20_cgroup_recovery_"
        "common_envelope_v1"
    ),
    "source_request_memory_mb": SOURCE_REQUEST_MEMORY_MB,
    "source_scheduler_memory_usage_mb": 9_766,
    "source_failure_class": "cgroup_memory_limit_exceeded_v1",
}

REQUIRED_ROUTE_SOURCE_PATHS = (
    "pipelines/exact_bench/noise_oracle_defaults.py",
    "pipelines/exact_bench/noise_oracle_runtime.py",
    "pipelines/scaffold/hh_continuation_scoring.py",
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/hh_backend_compile_oracle.py",
    "pipelines/static_adapt/ra_adapt/__init__.py",
    "pipelines/static_adapt/ra_adapt/adapters.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/phase0.py",
    "pipelines/static_adapt/ra_adapt/pools.py",
    "pipelines/static_adapt/ra_adapt/pure_hubbard_noise_page12.py",
    "pipelines/static_adapt/sr_snake/_selection.py",
)

PACKAGE_MANIFEST_SCHEMA = "paper_i_pure_hubbard_page12_noise_package_manifest_v1"
JOB_SCHEMA = "paper_i_pure_hubbard_page12_noise_job_v1"
AUTHORIZATION_SCHEMA = "paper_i_pure_hubbard_page12_noise_authorization_v1"
ACTIVATION_REQUEST_SCHEMA = "paper_i_pure_hubbard_page12_noise_activation_request_v1"
ACTIVATION_MANIFEST_SCHEMA = "paper_i_pure_hubbard_page12_noise_activation_manifest_v1"
EXECUTION_PLAN_SCHEMA = "paper_i_pure_hubbard_page12_noise_execution_plan_v1"
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_pure_hubbard_page12_noise_source_archive_manifest_v1"
)
P3_RECEIPT_SCHEMA = "paper_i_pure_hubbard_page12_noise_p3_receipt_v1"
P4_RECEIPT_SCHEMA = "paper_i_pure_hubbard_page12_noise_p4_receipt_v1"

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_numerical_preflight.py",
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
    "p3_numerical_receipt.json",
    "p4_packaged_numerical_receipt.json",
    "package_manifest.json",
)


class PackageContractError(RuntimeError):
    """Fail-closed package or worker-contract violation."""


def u_label(u_value: float) -> str:
    value = float(u_value)
    if value == 1.5:
        return "u1p5"
    if value == 8.0:
        return "u8"
    raise PackageContractError(f"Unsupported U/t value: {u_value!r}.")


def source_lock_id(u_value: float, noise_level: str) -> str:
    return f"pure_hubbard__{u_label(u_value)}__{noise_level}__page12_fullnoise_v1"


def execution_id(u_value: float, noise_level: str) -> str:
    return f"{EXECUTION_ID_PREFIX}__{u_label(u_value)}__{noise_level}"


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(u_value, level) for u_value, level, _ in CELL_ROWS)


def reject_cache_artifacts(root: Path) -> None:
    forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    if forbidden:
        raise PackageContractError(
            "Package contains forbidden Python cache artifacts: "
            + ", ".join(forbidden)
        )


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


def validate_control_file_bindings(
    package_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Verify the manifest's exact closed inventory of control-plane bytes."""

    raw_rows = manifest.get("control_files")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(CONTROL_FILES):
        raise PackageContractError("Control-file binding closure is absent.")
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Control-file binding {index} must be a mapping."
            )
        relative = safe_relative_path(
            raw.get("path"), label=f"control-file binding {index}"
        ).as_posix()
        if relative in rows:
            raise PackageContractError(
                f"Duplicate control-file binding: {relative}."
            )
        rows[relative] = dict(raw)
    if set(rows) != set(CONTROL_FILES):
        raise PackageContractError("Control-file path closure drifted.")
    root = package_root.resolve()
    for relative in CONTROL_FILES:
        row = rows[relative]
        path = package_root / relative
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise PackageContractError(
                f"Control file escaped the package: {relative}."
            ) from exc
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise PackageContractError(
                f"Control-file byte binding drifted: {relative}."
            )
    return {relative: rows[relative] for relative in CONTROL_FILES}


def repo_root_from_script(script: str | Path) -> Path:
    current = Path(script).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").is_file() and (
            candidate / "pipelines/static_adapt"
        ).is_dir():
            return candidate
    raise PackageContractError("Could not resolve the active repository root.")
