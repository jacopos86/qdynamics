#!/usr/bin/env python3
"""Run the two-arm strong--weak append-endpoint Phase-0 diagnostic to k=5."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import psutil

MODE_FIXED24_SHADOW = "fixed24_graph_weighted_adaptive_shadow_v1"
MODE_ACTIVE_ADAPTIVE = "active_adaptive_graph_weighted_v1"
ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY = (
    "native_phase0_proxy_cost_adaptive_shortlist_pending_semantic_closure"
)
ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA = (
    "paper_i_native_phase0_proxy_cost_adaptive_shortlist_receipt_pending"
)

# This scaffold must remain inert until the native semantic-closure
# implementation supplies these versioned identities and their exact digests.
NATIVE_SEMANTIC_IMPLEMENTATION_VERSION: str | None = None
NATIVE_ROUTE_IDS_BY_VARIANT: Mapping[str, str | None] = {
    "gradient_only": None,
    "proxy_cost": None,
}
NATIVE_ROUTE_DIGESTS_BY_VARIANT: Mapping[str, str | None] = {
    "gradient_only": None,
    "proxy_cost": None,
}
REQUIRED_NATIVE_SEMANTICS = {
    "phase0_variants": {
        "gradient_only": {
            "population": "same_append_endpoint_generator_population",
            "ranking_signal": "absolute_gradient",
            "structural_proxy_active": False,
            "filesystem_metric_active": False,
            "qiskit_active": False,
        },
        "proxy_cost": {
            "population": "same_append_endpoint_generator_population",
            "ranking_signal": (
                "absolute_gradient_weighted_by_existing_structural_proxy_transform"
            ),
            "structural_proxy_active": True,
            "filesystem_metric_active": False,
            "qiskit_active": False,
        },
    },
    "compile_scope": "phase0_proxy_or_off",
    "qiskit_active_phases": ["phase_i", "phase_ii", "phase_iii"],
    "compile_ansatz_scope": (
        "full_base_and_trial_ansatz_at_recorded_insertion_position"
    ),
    "signed_compile_deltas": ["dN2q", "dD2q", "dN1q"],
    "signed_delta_transform": "zero_centered_signed_arctan_v1",
    "negative_cancellation_rewarded": True,
    "selection_factor_active_phases": ["phase_i", "phase_ii", "phase_iii"],
    "s_alg_includes_compile_work": False,
    "all_other_route_semantics_identical": True,
}
EXECUTION_SURFACE_ENABLED = False


RUNNER_PATH = Path(__file__).resolve()
REPAIR_ROOT = RUNNER_PATH.parent
REPO_ROOT = RUNNER_PATH.parents[2]
PACKAGE_DIR = REPAIR_ROOT / (
    "paper_i_ra_adapt_page12_insertion_comparators_r50_20260812_v1_chtc"
)
SOURCE_EXECUTION_ID = (
    "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__"
    "strong_weak_u8__nph3__ra_global_singleton_gradient_phase0_"
    "phase123_qiskit_phase23_always_commutation_reduced"
)
SOURCE_JOB = PACKAGE_DIR / "jobs" / f"{SOURCE_EXECUTION_ID}.json"
CAMPAIGN_ID = (
    "paper_i_adaptive_append_endpoint_phase0_sw_always_k5_pair_20260816_v1"
)
AUTHORITY_DIR = REPAIR_ROOT / f"{CAMPAIGN_ID}_authority"
PLAN_PATH = AUTHORITY_DIR / "plan.json"
AUTHORIZATION_PATH = AUTHORITY_DIR / "authorization.json"
RUNTIME_ROOT = REPO_ROOT / "output/local_runs" / CAMPAIGN_ID
RUNS_ROOT = RUNTIME_ROOT / "runs"
STAGING_ROOT = RUNTIME_ROOT / "in_progress"
RECEIPTS_ROOT = RUNTIME_ROOT / "receipts"
GUARD_RECEIPTS_ROOT = RUNTIME_ROOT / "guard_receipts"
CAPACITY_RECEIPT_PATH = RUNTIME_ROOT / "capacity_receipt.json"
COMPARISON_PATH = RUNTIME_ROOT / "comparison_receipt.json"
TERMINAL_PATH = RUNTIME_ROOT / "terminal_receipt.json"
STATUS_PATH = RUNTIME_ROOT / "status.json"
LOCK_PATH = RUNTIME_ROOT / "campaign.lock"
TARGET_HORIZON = 5
SHORTLIST_CAP = 24
RSS_LIMIT_BYTES = 8 * 1024**3
AVAILABLE_MEMORY_FLOOR_BYTES = 2 * 1024**3
MIN_LAUNCH_AVAILABLE_MEMORY_BYTES = 5 * 1024**3
RUNTIME_FREE_DISK_FLOOR_BYTES = 2 * 1024**3
MIN_LAUNCH_FREE_DISK_BYTES = 10 * 1024**3
MAXIMUM_CAPACITY_WAIT_SECONDS = 24 * 60 * 60
CAPACITY_POLL_SECONDS = 10.0
CHILD_POLL_SECONDS = 1.0
STATUS_SECONDS = 10.0
CHILD_TOKEN_ENV = "PAPER_I_ADAPTIVE_APPEND_ENDPOINT_PHASE0_CHILD_TOKEN"

PLAN_SCHEMA = "paper_i_adaptive_append_endpoint_phase0_pair_plan_v1"
AUTHORIZATION_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_pair_authorization_v1"
)
CAPACITY_SCHEMA = "paper_i_adaptive_append_endpoint_phase0_capacity_v1"
ROUTE_OVERLAY_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_route_overlay_v1"
)
EXECUTION_MANIFEST_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_execution_manifest_v1"
)
WORKER_RECEIPT_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_worker_receipt_v1"
)
GUARD_RECEIPT_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_guard_receipt_v1"
)
COMPARISON_SCHEMA = (
    "paper_i_adaptive_append_endpoint_phase0_terminal_comparison_v1"
)
TERMINAL_SCHEMA = "paper_i_adaptive_append_endpoint_phase0_terminal_receipt_v1"
STATUS_SCHEMA = "paper_i_adaptive_append_endpoint_phase0_status_v1"

EXPECTED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "CUDA_VISIBLE_DEVICES": "",
    "STATIC_ADAPT_HH_POOL_CACHE": "off",
    "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "off",
}


class RunnerError(RuntimeError):
    """Raised when the diagnostic cannot preserve its exact contract."""


def assert_native_route_ready() -> None:
    """Refuse execution until both versioned native routes are fully bound."""

    route_ids = tuple(NATIVE_ROUTE_IDS_BY_VARIANT.values())
    digests = tuple(NATIVE_ROUTE_DIGESTS_BY_VARIANT.values())
    ready = bool(
        EXECUTION_SURFACE_ENABLED
        and NATIVE_SEMANTIC_IMPLEMENTATION_VERSION
        and all(isinstance(value, str) and value for value in route_ids)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in digests
        )
    )
    if not ready:
        raise RunnerError(
            "Native Phase-0 semantic closure is unresolved: execution and "
            "authority materialization remain disabled until the two versioned "
            "route IDs, exact route digests, and implementation version are bound."
        )


@dataclass(frozen=True)
class CellSpec:
    mode: str
    execution_id: str
    regime_id: str = "strong_weak_u8"
    nph: int = 3
    target_horizon: int = TARGET_HORIZON
    insertion_policy: str = "always_commutation_reduced"
    fresh_start: bool = True
    submission_authorized: bool = False
    paper_adoption_authorized: bool = False
    paper_evidence_adoption_authorized: bool = False


CELL_SPECS = (
    CellSpec(
        mode=MODE_FIXED24_SHADOW,
        execution_id=(
            "adaptive_append_endpoint_phase0__strong_weak_u8__nph3__"
            "ra_always_commutation_reduced__fixed24_shadow__k5"
        ),
    ),
    CellSpec(
        mode=MODE_ACTIVE_ADAPTIVE,
        execution_id=(
            "adaptive_append_endpoint_phase0__strong_weak_u8__nph3__"
            "ra_always_commutation_reduced__active_adaptive__k5"
        ),
    ),
)
if len({cell.mode for cell in CELL_SPECS}) != 2 or len(
    {cell.execution_id for cell in CELL_SPECS}
) != 2:
    raise RuntimeError("Adaptive Phase-0 pair contains a duplicate arm.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise RunnerError("Self-digested payload already contains sha256.")
    result["sha256"] = canonical_sha256(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RunnerError(f"Required regular file is absent: {path}")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Could not load JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RunnerError(f"JSON payload is not a mapping: {path}")
    return payload


def load_digested(path: Path, *, schema: str) -> dict[str, Any]:
    payload = load_json(path)
    observed = payload.pop("sha256", None)
    if payload.get("schema") != schema or observed != canonical_sha256(payload):
        raise RunnerError(f"Self-digested payload drifted: {path}")
    payload["sha256"] = observed
    return payload


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(canonical_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_worker() -> Any:
    path = PACKAGE_DIR / "run_cell.py"
    spec = importlib.util.spec_from_file_location(
        "paper_i_adaptive_append_phase0_parent_worker",
        path,
    )
    if spec is None or spec.loader is None:
        raise RunnerError("Could not load the source-locked parent worker.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_bindings() -> dict[str, Any]:
    job = load_json(SOURCE_JOB)
    protocol_path = PACKAGE_DIR / Path(str(job["protocol_path"]))
    source_archive = PACKAGE_DIR / "source/source_locked.tar.gz"
    if (
        job.get("execution_id") != SOURCE_EXECUTION_ID
        or job.get("regime_id") != "strong_weak_u8"
        or job.get("nph") != 3
        or job.get("comparator_policy") != "always_commutation_reduced"
        or job.get("runtime_insertion_mode") != "full_commutation_reduced"
        or job.get("typed_insertion_kind") != "always_commutation_reduced"
        or job.get("target_horizon") != 50
        or job.get("protocol_file_sha256") != sha256_file(protocol_path)
        or job.get("sha256")
        != canonical_sha256(
            {key: value for key, value in job.items() if key != "sha256"}
        )
    ):
        raise RunnerError("Parent strong--weak always-open job identity drifted.")
    protocol = load_json(protocol_path)
    if (
        protocol.get("sha256") != job.get("protocol_sha256")
        or protocol.get("horizon") != 50
        or protocol.get("request", {})
        .get("method", {})
        .get("insertion", {})
        .get("kind")
        != "always_commutation_reduced"
        or protocol.get("request", {})
        .get("execution", {})
        .get("stop", {})
        .get("maximum_controller_rounds")
        != 50
    ):
        raise RunnerError("Parent strong--weak always-open protocol drifted.")
    archive_binding = file_binding(source_archive)
    if archive_binding["sha256"] != (
        "690d54dbf5bafcaaf974dc11339ed927cb7f5d117265ed51adbb811785740762"
    ):
        raise RunnerError("Parent source archive drifted.")
    if (
        str(job["protocol_sha256"])
        != "37d77f48342cf29f70bcb9710840be0e4a4b7e7d2aac28e8e4dd0cad559064f1"
        or str(job["route_contract_sha256"])
        != "24d5aed82ee202293187deb5e9745875a5779f8d6bca806536e4a323c7a307a6"
    ):
        raise RunnerError("Exact parent protocol or route binding drifted.")
    return {
        "job": {**file_binding(SOURCE_JOB), "canonical_sha256": job["sha256"]},
        "protocol": {
            **file_binding(protocol_path),
            "canonical_sha256": protocol["sha256"],
        },
        "source_archive": archive_binding,
        "parent_route_contract_sha256": str(job["route_contract_sha256"]),
        "parent_protocol_sha256": str(job["protocol_sha256"]),
    }


def build_plan() -> dict[str, Any]:
    return digested(
        {
            "schema": PLAN_SCHEMA,
            "created_at": utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "run_class": "diagnostic",
            "method": "RA-ADAPT always-open commutation-reduced insertion",
            "phase0_policy": ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY,
            "phase0_receipt_schema": (
                ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA
            ),
            "phase0_scoring_domain": "append_endpoint_generator_gradient_v1",
            "phase0_graph_proxy_weighting": True,
            "phase0_shortlist_cap": SHORTLIST_CAP,
            "required_native_semantics": REQUIRED_NATIVE_SEMANTICS,
            "native_semantic_implementation_version": (
                NATIVE_SEMANTIC_IMPLEMENTATION_VERSION
            ),
            "native_route_ids_by_variant": dict(NATIVE_ROUTE_IDS_BY_VARIANT),
            "native_route_digests_by_variant": dict(
                NATIVE_ROUTE_DIGESTS_BY_VARIANT
            ),
            "selected_native_phase0_variant": "proxy_cost",
            "cells": [asdict(cell) for cell in CELL_SPECS],
            "fixed_execution_order": [cell.mode for cell in CELL_SPECS],
            "source": source_bindings(),
            "runner": file_binding(RUNNER_PATH),
            "wrapper_used": False,
            "wrapper_kind": None,
            "settings_reused": [
                "strong_weak_u8_same_cutoff_problem_and_exact_reference",
                "nph3_global_guarded_singleton_pool",
                "always_commutation_reduced_later_insertion_policy",
                "phase1_phase2_phase3_qiskit_selector",
                "stationary_source_response_v1",
                "powell_maxiter_200",
                "adapt_and_transpiler_seed_7",
            ],
            "settings_changed": [
                "phase0_ranking:absolute_gradient_to_graph_weighted_utility",
                "phase0_shortlist:fixed24_shadow_then_active_adaptive",
                "maximum_controller_rounds:50_to_5",
                "native_phase0_semantics:pending_versioned_route_binding",
            ],
            "unresolved_source_fields": [
                "native_semantic_implementation_version",
                "native_gradient_only_route_id",
                "native_gradient_only_route_digest",
                "native_proxy_cost_route_id",
                "native_proxy_cost_route_digest",
            ],
            "fresh_start": True,
            "maximum_concurrency": 1,
            "execution_target": "local_mac_guarded_serial_cpu_only",
            "coordination_scope": "dedicated_campaign_lock_and_capacity_only",
            "capacity_basis": {
                "reference_campaign": (
                    "paper_i_position_aware_phase0_sw_always_k15_20260816_v1"
                ),
                "reference_elapsed_seconds": 327.5,
                "reference_peak_rss_bytes": 1_944_420_352,
                "launch_available_memory_floor_bytes": (
                    MIN_LAUNCH_AVAILABLE_MEMORY_BYTES
                ),
                "rss_limit_bytes": RSS_LIMIT_BYTES,
                "runtime_available_memory_floor_bytes": (
                    AVAILABLE_MEMORY_FLOOR_BYTES
                ),
                "launch_free_disk_floor_bytes": MIN_LAUNCH_FREE_DISK_BYTES,
            },
            "maximum_capacity_wait_seconds": MAXIMUM_CAPACITY_WAIT_SECONDS,
            "runtime_environment": dict(EXPECTED_ENV),
            "execution_surface_enabled": False,
            "execution_authorized": False,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def materialize_authority() -> dict[str, Any]:
    assert_native_route_ready()
    if AUTHORITY_DIR.exists() or AUTHORITY_DIR.is_symlink():
        raise RunnerError(f"Authority already exists: {AUTHORITY_DIR}")
    AUTHORITY_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{AUTHORITY_DIR.name}.", dir=AUTHORITY_DIR.parent)
    )
    try:
        plan = build_plan()
        write_json_exclusive(temporary / "plan.json", plan)
        authorization = digested(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "created_at": utc_now(),
                "authorization_kind": (
                    "explicit_current_user_two_arm_diagnostic_execution"
                ),
                "scope": (
                    "two_serial_local_strong_weak_append_endpoint_phase0_"
                    "always_open_k5_cells"
                ),
                "campaign_id": CAMPAIGN_ID,
                "modes": [cell.mode for cell in CELL_SPECS],
                "execution_ids": [cell.execution_id for cell in CELL_SPECS],
                "plan_sha256": plan["sha256"],
                "runner_sha256": plan["runner"]["sha256"],
                "target_horizon": TARGET_HORIZON,
                "execution_authorized": True,
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(temporary / "authorization.json", authorization)
        os.rename(temporary, AUTHORITY_DIR)
        return authorization
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    assert_native_route_ready()
    plan = load_digested(PLAN_PATH, schema=PLAN_SCHEMA)
    authorization = load_digested(
        AUTHORIZATION_PATH,
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("cells") != [asdict(cell) for cell in CELL_SPECS]
        or plan.get("fixed_execution_order") != [cell.mode for cell in CELL_SPECS]
        or plan.get("source") != source_bindings()
        or plan.get("runner") != file_binding(RUNNER_PATH)
        or plan.get("coordination_scope")
        != "dedicated_campaign_lock_and_capacity_only"
        or plan.get("execution_authorized") is not False
        or plan.get("submission_authorized") is not False
        or plan.get("paper_adoption_authorized") is not False
        or plan.get("paper_evidence_adoption_authorized") is not False
        or authorization.get("campaign_id") != CAMPAIGN_ID
        or authorization.get("modes") != [cell.mode for cell in CELL_SPECS]
        or authorization.get("execution_ids")
        != [cell.execution_id for cell in CELL_SPECS]
        or authorization.get("plan_sha256") != plan["sha256"]
        or authorization.get("runner_sha256") != plan["runner"]["sha256"]
        or authorization.get("target_horizon") != TARGET_HORIZON
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not False
        or authorization.get("paper_adoption_authorized") is not False
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise RunnerError("Adaptive Phase-0 pair authority chain drifted.")
    return plan, authorization


def assert_environment() -> None:
    drift = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in EXPECTED_ENV.items()
        if os.environ.get(key) != value
    }
    if drift:
        raise RunnerError(f"Numerical CPU-only environment drifted: {drift}")


def capacity_snapshot(
    *,
    available_memory_bytes: int,
    free_disk_bytes: int,
) -> dict[str, Any]:
    memory = int(available_memory_bytes)
    disk = int(free_disk_bytes)
    ready = bool(
        memory >= MIN_LAUNCH_AVAILABLE_MEMORY_BYTES
        and disk >= MIN_LAUNCH_FREE_DISK_BYTES
    )
    return {
        "schema": CAPACITY_SCHEMA,
        "status": (
            "ready_for_adaptive_pair"
            if ready
            else "waiting_for_launch_capacity"
        ),
        "observed_at": utc_now(),
        "available_memory_bytes": memory,
        "minimum_launch_available_memory_bytes": (
            MIN_LAUNCH_AVAILABLE_MEMORY_BYTES
        ),
        "free_disk_bytes": disk,
        "minimum_launch_free_disk_bytes": MIN_LAUNCH_FREE_DISK_BYTES,
        "launch_ready": ready,
        "coordination_scope": "dedicated_campaign_lock_and_capacity_only",
        "scientific_execution_performed": False,
    }


def wait_for_launch_capacity(
    *,
    maximum_wait_seconds: float = MAXIMUM_CAPACITY_WAIT_SECONDS,
    poll_seconds: float = CAPACITY_POLL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    memory_supplier: Callable[[], int] = lambda: int(
        psutil.virtual_memory().available
    ),
    disk_supplier: Callable[[], int] = lambda: int(
        shutil.disk_usage(REPO_ROOT).free
    ),
    status_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if maximum_wait_seconds <= 0.0 or poll_seconds < 0.0:
        raise RunnerError("Capacity-wait bounds must be finite and nonnegative.")
    started = clock()
    while True:
        now = clock()
        elapsed = now - started
        snapshot = capacity_snapshot(
            available_memory_bytes=memory_supplier(),
            free_disk_bytes=disk_supplier(),
        )
        snapshot["elapsed_wait_seconds"] = elapsed
        snapshot["maximum_wait_seconds"] = maximum_wait_seconds
        if snapshot["launch_ready"]:
            if status_sink is not None:
                status_sink(dict(snapshot))
            return snapshot
        if elapsed >= maximum_wait_seconds:
            failure = dict(snapshot)
            failure.update(
                {
                    "status": "failed_capacity_wait_timeout",
                    "launch_ready": False,
                    "terminal_failure": True,
                }
            )
            if status_sink is not None:
                status_sink(failure)
            raise RunnerError("The bounded capacity wait timed out.")
        if status_sink is not None:
            status_sink(dict(snapshot))
        sleeper(poll_seconds)


@contextmanager
def campaign_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(
                "The adaptive Phase-0 pair already owns its lock."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def assert_pristine_cell_paths(
    run_dir: Path,
    staging_dir: Path,
    receipt_path: Path,
) -> None:
    occupied = [
        str(path)
        for path in (run_dir, staging_dir, receipt_path)
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise RunnerError(f"Adaptive diagnostic cell paths are not pristine: {occupied}")


def _cell_by_mode(mode: str) -> CellSpec:
    matches = [cell for cell in CELL_SPECS if cell.mode == mode]
    if len(matches) != 1:
        raise RunnerError(f"Unknown or duplicate adaptive diagnostic mode: {mode}")
    return matches[0]


def _cell_paths(cell: CellSpec) -> tuple[Path, Path, Path, Path]:
    return (
        RUNS_ROOT / cell.execution_id,
        STAGING_ROOT / cell.execution_id,
        RECEIPTS_ROOT / f"{cell.execution_id}.json",
        GUARD_RECEIPTS_ROOT / f"{cell.execution_id}.json",
    )


def child_token(authorization_sha256: str, cell: CellSpec) -> str:
    return hashlib.sha256(
        (
            f"{authorization_sha256}:{sha256_file(RUNNER_PATH)}:"
            f"{cell.execution_id}:{cell.mode}"
        ).encode()
    ).hexdigest()


def _activate_native_route(_mode: str) -> Callable[[], None]:
    raise RunnerError(
        "Native Phase-0 runtime activation has not been integrated. The "
        "temporary monkeypatch overlay is intentionally not an execution path."
    )


def _summary_terminal_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    trace = summary.get("accepted_error_trace")
    work = summary.get("canonical_all_work")
    if (
        summary.get("available_controller_rounds") != TARGET_HORIZON
        or not isinstance(trace, list)
        or len(trace) != TARGET_HORIZON
        or not isinstance(work, Mapping)
        or not isinstance(work.get("components"), Mapping)
    ):
        raise RunnerError("Cell summary does not close the exact k=5 horizon.")
    if [row.get("controller_round") for row in trace] != list(
        range(1, TARGET_HORIZON + 1)
    ):
        raise RunnerError("Cell summary controller-round sequence drifted.")
    references = {float(row["exact_same_cutoff_energy"]) for row in trace}
    final = trace[-1]
    components = {
        key: int(work["components"][key])
        for key in ("n_h_outer", "n_h_refit", "n_grad", "n_metric")
    }
    s_alg = int(work["s_alg"])
    values = (
        float(final["accepted_energy"]),
        float(final["absolute_energy_error"]),
        *references,
    )
    if (
        len(references) != 1
        or not all(math.isfinite(value) for value in values)
        or any(value < 0 for value in components.values())
        or sum(components.values()) != s_alg
    ):
        raise RunnerError("Cell summary energy or S_alg closure failed.")
    return {
        "controller_round": TARGET_HORIZON,
        "accepted_energy": float(final["accepted_energy"]),
        "absolute_energy_error": float(final["absolute_energy_error"]),
        "exact_same_cutoff_energy": references.pop(),
        "active_ansatz_depth": int(final["active_ansatz_depth"]),
        "s_alg": s_alg,
        "s_alg_components": components,
    }


def build_terminal_comparison(
    *,
    summaries: Mapping[str, Mapping[str, Any]],
    worker_receipt_sha256_by_mode: Mapping[str, str],
    guard_receipt_sha256_by_mode: Mapping[str, str],
    capacity_receipt_sha256: str,
) -> dict[str, Any]:
    modes = [cell.mode for cell in CELL_SPECS]
    if (
        set(summaries) != set(modes)
        or set(worker_receipt_sha256_by_mode) != set(modes)
        or set(guard_receipt_sha256_by_mode) != set(modes)
        or any(
            len(value) != 64
            for value in (
                capacity_receipt_sha256,
                *worker_receipt_sha256_by_mode.values(),
                *guard_receipt_sha256_by_mode.values(),
            )
        )
    ):
        raise RunnerError("Terminal comparison input identity is incomplete.")
    rows: list[dict[str, Any]] = []
    for cell in CELL_SPECS:
        row = _summary_terminal_row(summaries[cell.mode])
        rows.append(
            {
                "mode": cell.mode,
                "execution_id": cell.execution_id,
                **row,
                "worker_receipt_sha256": worker_receipt_sha256_by_mode[
                    cell.mode
                ],
                "guard_receipt_sha256": guard_receipt_sha256_by_mode[cell.mode],
            }
        )
    if rows[0]["exact_same_cutoff_energy"] != rows[1][
        "exact_same_cutoff_energy"
    ]:
        raise RunnerError("The pair does not share one same-cutoff exact reference.")
    fixed, adaptive = rows
    return digested(
        {
            "schema": COMPARISON_SCHEMA,
            "status": "passed_exact_two_cells_k5",
            "campaign_id": CAMPAIGN_ID,
            "regime_id": "strong_weak_u8",
            "nph": 3,
            "target_horizon": TARGET_HORIZON,
            "insertion_policy": "always_commutation_reduced",
            "fixed_execution_order": modes,
            "cells": rows,
            "comparison": {
                "direction": "active_adaptive_minus_fixed24_shadow",
                "adaptive_minus_fixed": {
                    "accepted_energy": (
                        adaptive["accepted_energy"] - fixed["accepted_energy"]
                    ),
                    "absolute_energy_error": (
                        adaptive["absolute_energy_error"]
                        - fixed["absolute_energy_error"]
                    ),
                    "s_alg": adaptive["s_alg"] - fixed["s_alg"],
                    "s_alg_components": {
                        key: (
                            adaptive["s_alg_components"][key]
                            - fixed["s_alg_components"][key]
                        )
                        for key in fixed["s_alg_components"]
                    },
                },
            },
            "capacity_receipt_sha256": capacity_receipt_sha256,
            "run_class": "diagnostic",
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )


def _write_child_outputs(
    *,
    result: Any,
    rounds: int,
    cell: CellSpec,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    staging: Path,
) -> dict[str, Any]:
    result_payload = result.to_dict()
    if result.run.paper_i_summary is None:
        raise RunnerError("Adaptive diagnostic omitted its Paper-I summary.")
    summary = result.run.paper_i_summary.to_dict()
    _summary_terminal_row(summary)
    write_json_exclusive(staging / "result/result.json", result_payload)
    write_json_exclusive(staging / "summary/summary.json", summary)
    route_overlay = digested(
        {
            "schema": ROUTE_OVERLAY_SCHEMA,
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "mode": cell.mode,
            "parent_execution_id": SOURCE_EXECUTION_ID,
            "parent_protocol_sha256": plan["source"]["parent_protocol_sha256"],
            "parent_route_contract_sha256": plan["source"][
                "parent_route_contract_sha256"
            ],
            "runner_sha256": plan["runner"]["sha256"],
            "native_semantic_implementation_version": (
                plan["native_semantic_implementation_version"]
            ),
            "native_route_id": plan["native_route_ids_by_variant"]["proxy_cost"],
            "native_route_digest": plan["native_route_digests_by_variant"][
                "proxy_cost"
            ],
            "phase0_policy": ADAPTIVE_APPEND_ENDPOINT_PHASE0_POLICY,
            "phase0_receipt_schema": (
                ADAPTIVE_APPEND_ENDPOINT_PHASE0_RECEIPT_SCHEMA
            ),
            "phase0_scoring_domain": "append_endpoint_generator_gradient_v1",
            "phase0_graph_proxy_weighting": True,
            "target_horizon": TARGET_HORIZON,
            "later_insertion_policy": "always_commutation_reduced",
            "fresh_start": True,
            "run_class": "diagnostic",
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "route_overlay.json", route_overlay)
    artifacts: dict[str, dict[str, Any]] = {}
    for role, relative in {
        "checkpoint": "checkpoints/current.json",
        "estimator_ledger": "result/estimator_ledger.json",
        "result": "result/result.json",
        "summary": "summary/summary.json",
        "route_overlay": "route_overlay.json",
    }.items():
        path = staging / relative
        if not path.is_file():
            raise RunnerError(f"Adaptive output role is absent: {role}")
        artifacts[role] = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    manifest = digested(
        {
            "schema": EXECUTION_MANIFEST_SCHEMA,
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "mode": cell.mode,
            "parent_execution_id": SOURCE_EXECUTION_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "parent_protocol_sha256": plan["source"]["parent_protocol_sha256"],
            "target_horizon": TARGET_HORIZON,
            "controller_rounds_completed": int(rounds),
            "fresh_start": True,
            "source_checkpoint_consumed": False,
            "source_result_consumed": False,
            "resume_claimed": False,
            "run_class": "diagnostic",
            "artifacts": artifacts,
            "submission_authorized": False,
            "paper_adoption_authorized": False,
            "paper_evidence_adoption_authorized": False,
        }
    )
    write_json_exclusive(staging / "execution_manifest.json", manifest)
    return manifest


def run_child(mode: str) -> int:
    cell = _cell_by_mode(mode)
    plan, authorization = validate_authority()
    assert_environment()
    if os.environ.get(CHILD_TOKEN_ENV) != child_token(
        authorization["sha256"], cell
    ):
        raise RunnerError("Adaptive diagnostic child capability is invalid.")
    run_dir, staging, receipt_path, _guard_path = _cell_paths(cell)
    assert_pristine_cell_paths(run_dir, staging, receipt_path)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    worker = import_worker()
    temporary = None
    restore_native_route: Callable[[], None] | None = None
    try:
        job, _manifest, protocol, problem, temporary = worker._prepare(SOURCE_JOB)
        if (
            job.get("execution_id") != SOURCE_EXECUTION_ID
            or protocol.request.method.insertion.kind
            != "always_commutation_reduced"
            or int(protocol.horizon) != 50
        ):
            raise RunnerError("Prepared parent protocol drifted.")
        restore_native_route = _activate_native_route(cell.mode)
        source_root = Path(temporary.name) / "source"
        original = Path.cwd()
        os.chdir(source_root)
        try:
            result, rounds = worker._execute(
                protocol=protocol,
                problem=problem,
                staging=staging,
                maximum_rounds=TARGET_HORIZON,
            )
        finally:
            os.chdir(original)
        if rounds != TARGET_HORIZON:
            raise RunnerError(
                f"Adaptive arm {cell.mode} stopped at {rounds}, not k=5."
            )
        execution_manifest = _write_child_outputs(
            result=result,
            rounds=rounds,
            cell=cell,
            plan=plan,
            authorization=authorization,
            staging=staging,
        )
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, run_dir)
        receipt = digested(
            {
                "schema": WORKER_RECEIPT_SCHEMA,
                "status": "passed_k5",
                "campaign_id": CAMPAIGN_ID,
                "execution_id": cell.execution_id,
                "mode": cell.mode,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "execution_manifest_sha256": execution_manifest["sha256"],
                "controller_rounds_completed": rounds,
                "fresh_start": True,
                "artifacts": [
                    {
                        "path": path.relative_to(RUNTIME_ROOT).as_posix(),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(run_dir.rglob("*"))
                    if path.is_file()
                ],
                "run_class": "diagnostic",
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(receipt_path, receipt)
        return 0
    finally:
        if restore_native_route is not None:
            restore_native_route()
        if temporary is not None:
            temporary.cleanup()


def total_rss(process: psutil.Process) -> int:
    total = 0
    for candidate in [process, *process.children(recursive=True)]:
        try:
            total += int(candidate.memory_info().rss)
        except psutil.Error:
            pass
    return total


def _write_status(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body.setdefault("schema", STATUS_SCHEMA)
    body.setdefault("updated_at", utc_now())
    body.setdefault("campaign_id", CAMPAIGN_ID)
    write_json_atomic(STATUS_PATH, digested(body))


def _monitor_cell(
    *,
    cell: CellSpec,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment[CHILD_TOKEN_ENV] = child_token(authorization["sha256"], cell)
    command = [
        sys.executable,
        "-u",
        "-B",
        str(RUNNER_PATH),
        "--child-mode",
        cell.mode,
    ]
    child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment)
    process = psutil.Process(child.pid)
    started = time.monotonic()
    last_status = 0.0
    peak_rss = 0
    minimum_available = int(psutil.virtual_memory().available)
    minimum_disk = int(shutil.disk_usage(REPO_ROOT).free)
    stop_reason: str | None = None
    try:
        while child.poll() is None:
            rss = total_rss(process)
            available = int(psutil.virtual_memory().available)
            free_disk = int(shutil.disk_usage(REPO_ROOT).free)
            peak_rss = max(peak_rss, rss)
            minimum_available = min(minimum_available, available)
            minimum_disk = min(minimum_disk, free_disk)
            if rss > RSS_LIMIT_BYTES:
                stop_reason = "rss_limit_breached"
            elif available < AVAILABLE_MEMORY_FLOOR_BYTES:
                stop_reason = "available_memory_floor_breached"
            elif free_disk < RUNTIME_FREE_DISK_FLOOR_BYTES:
                stop_reason = "free_disk_floor_breached"
            now = time.monotonic()
            if now - last_status >= STATUS_SECONDS:
                _write_status(
                    {
                        "status": "running_adaptive_phase0_k5_cell",
                        "mode": cell.mode,
                        "execution_id": cell.execution_id,
                        "cell_index": CELL_SPECS.index(cell) + 1,
                        "cell_count": len(CELL_SPECS),
                        "child_pid": child.pid,
                        "elapsed_seconds": now - started,
                        "current_rss_bytes": rss,
                        "peak_rss_bytes": peak_rss,
                        "available_memory_bytes": available,
                        "minimum_available_memory_bytes": minimum_available,
                        "free_disk_bytes": free_disk,
                        "minimum_free_disk_bytes": minimum_disk,
                        "stop_reason": stop_reason,
                    }
                )
                last_status = now
            if stop_reason is not None:
                child.terminate()
                break
            time.sleep(CHILD_POLL_SECONDS)
    except BaseException:
        if child.poll() is None:
            child.terminate()
        raise
    try:
        returncode = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        child.kill()
        returncode = child.wait(timeout=30)
    elapsed = time.monotonic() - started
    if returncode != 0 or stop_reason is not None:
        raise RunnerError(
            f"Adaptive arm {cell.mode} failed rc={returncode}, "
            f"reason={stop_reason}."
        )
    receipt_path = _cell_paths(cell)[2]
    worker_receipt = load_digested(
        receipt_path,
        schema=WORKER_RECEIPT_SCHEMA,
    )
    guard = digested(
        {
            "schema": GUARD_RECEIPT_SCHEMA,
            "status": "passed",
            "campaign_id": CAMPAIGN_ID,
            "execution_id": cell.execution_id,
            "mode": cell.mode,
            "child_returncode": returncode,
            "worker_receipt_sha256": worker_receipt["sha256"],
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
            "minimum_available_memory_bytes": minimum_available,
            "minimum_free_disk_bytes": minimum_disk,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "available_memory_floor_bytes": AVAILABLE_MEMORY_FLOOR_BYTES,
            "runtime_free_disk_floor_bytes": RUNTIME_FREE_DISK_FLOOR_BYTES,
            "stop_reason": None,
        }
    )
    write_json_exclusive(_cell_paths(cell)[3], guard)
    return guard


def terminal_is_valid() -> bool:
    if not TERMINAL_PATH.is_file():
        return False
    try:
        terminal = load_digested(TERMINAL_PATH, schema=TERMINAL_SCHEMA)
        comparison = load_digested(COMPARISON_PATH, schema=COMPARISON_SCHEMA)
    except RunnerError:
        return False
    return bool(
        terminal.get("status") == "passed_exact_two_cells_k5"
        and terminal.get("campaign_id") == CAMPAIGN_ID
        and terminal.get("comparison_receipt_sha256") == comparison["sha256"]
        and terminal.get("submission_authorized") is False
        and terminal.get("paper_adoption_authorized") is False
        and terminal.get("paper_evidence_adoption_authorized") is False
    )


def run_supervisor() -> int:
    plan, authorization = validate_authority()
    assert_environment()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    with campaign_lock(LOCK_PATH):
        if terminal_is_valid():
            return 0
        if any(
            path.exists() or path.is_symlink()
            for path in (CAPACITY_RECEIPT_PATH, COMPARISON_PATH, TERMINAL_PATH)
        ):
            raise RunnerError("Adaptive campaign has partial terminal state.")
        for cell in CELL_SPECS:
            assert_pristine_cell_paths(*_cell_paths(cell)[:3])

        def status_sink(snapshot: dict[str, Any]) -> None:
            _write_status(
                {
                    **snapshot,
                    "status": snapshot["status"],
                    "phase": "initial_capacity_wait",
                }
            )

        initial_capacity = wait_for_launch_capacity(status_sink=status_sink)
        capacity_receipt = digested(
            {
                **initial_capacity,
                "schema": CAPACITY_SCHEMA,
                "status": "passed_launch_capacity",
                "campaign_id": CAMPAIGN_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "scientific_execution_performed": False,
            }
        )
        write_json_exclusive(CAPACITY_RECEIPT_PATH, capacity_receipt)
        worker_sha_by_mode: dict[str, str] = {}
        guard_sha_by_mode: dict[str, str] = {}
        summaries: dict[str, Mapping[str, Any]] = {}
        for index, cell in enumerate(CELL_SPECS, start=1):
            wait_for_launch_capacity(
                status_sink=lambda snapshot, index=index, cell=cell: _write_status(
                    {
                        **snapshot,
                        "status": snapshot["status"],
                        "phase": "cell_capacity_wait",
                        "cell_index": index,
                        "cell_count": len(CELL_SPECS),
                        "mode": cell.mode,
                        "execution_id": cell.execution_id,
                    }
                )
            )
            guard = _monitor_cell(cell=cell, authorization=authorization)
            worker = load_digested(
                _cell_paths(cell)[2],
                schema=WORKER_RECEIPT_SCHEMA,
            )
            worker_sha_by_mode[cell.mode] = worker["sha256"]
            guard_sha_by_mode[cell.mode] = guard["sha256"]
            summary_path = _cell_paths(cell)[0] / "summary/summary.json"
            summaries[cell.mode] = load_json(summary_path)
        comparison = build_terminal_comparison(
            summaries=summaries,
            worker_receipt_sha256_by_mode=worker_sha_by_mode,
            guard_receipt_sha256_by_mode=guard_sha_by_mode,
            capacity_receipt_sha256=capacity_receipt["sha256"],
        )
        write_json_exclusive(COMPARISON_PATH, comparison)
        terminal = digested(
            {
                "schema": TERMINAL_SCHEMA,
                "status": "passed_exact_two_cells_k5",
                "campaign_id": CAMPAIGN_ID,
                "plan_sha256": plan["sha256"],
                "authorization_sha256": authorization["sha256"],
                "capacity_receipt_sha256": capacity_receipt["sha256"],
                "comparison_receipt_sha256": comparison["sha256"],
                "fixed_execution_order": [cell.mode for cell in CELL_SPECS],
                "controller_rounds_completed_by_cell": {
                    cell.mode: TARGET_HORIZON for cell in CELL_SPECS
                },
                "run_class": "diagnostic",
                "submission_authorized": False,
                "paper_adoption_authorized": False,
                "paper_evidence_adoption_authorized": False,
            }
        )
        write_json_exclusive(TERMINAL_PATH, terminal)
        _write_status(
            {
                "status": "passed_exact_two_cells_k5",
                "terminal_sha256": terminal["sha256"],
                "comparison_receipt_sha256": comparison["sha256"],
            }
        )
        return 0


def preflight() -> dict[str, Any]:
    plan, authorization = validate_authority()
    snapshot = capacity_snapshot(
        available_memory_bytes=int(psutil.virtual_memory().available),
        free_disk_bytes=int(shutil.disk_usage(REPO_ROOT).free),
    )
    return digested(
        {
            **snapshot,
            "schema": "paper_i_adaptive_append_endpoint_phase0_preflight_v1",
            "campaign_id": CAMPAIGN_ID,
            "plan_sha256": plan["sha256"],
            "authorization_sha256": authorization["sha256"],
            "terminal_state": (
                "complete"
                if terminal_is_valid()
                else "partial"
                if RUNTIME_ROOT.exists()
                and any(path.name != "campaign.lock" for path in RUNTIME_ROOT.iterdir())
                else "absent"
            ),
            "scientific_execution_performed": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-plan", action="store_true", required=True)
    args = parser.parse_args()
    if not args.show_plan:
        raise RunnerError("The inert scaffold exposes only --show-plan.")
    print(json.dumps(build_plan(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
