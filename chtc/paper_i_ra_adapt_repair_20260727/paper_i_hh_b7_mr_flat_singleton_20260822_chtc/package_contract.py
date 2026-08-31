#!/usr/bin/env python3
"""Closed contract for B7: flat-pool measured residual (manuscript route, stationarity off).

B7 executes the candidate traversal Paper I describes -- the fixed, deduplicated
global guarded-singleton pool (948 weak / 6508 strong Pauli words) screened
through Phases 0--III with identity exposure and NO in-pipeline splitting --
under the measured-residual non-stationary lane:
  algorithm_id            = paper_i_ra_adapt_nonstationary_incremental_full_response_v2
  active_gradient_policy  = measured_residual_response_v1
  phase3_candidate_gain   = joint_minus_active_only_supported_trust_v1
Request adapter: GlobalSinglePauliWordCandidateAdapter (phase-I supply = global
guarded singletons, phase-II exposure = retained singleton identities).

One axis vs B3 (stationarity: measured instead of stationary) and one axis vs
B5/b3mr (pool traversal: flat 948 children instead of ~123 macro parents with
post-shortlist splitting). No compile-cost term, no candidate warm start.
Matrix: 3 insertion policies x 6 regimes = 18 cells; seeds (adapt 7,
transpiler 7), cutoffs (weak nph3, strong nph7), horizons (weak 50, strong 75)
identical to b3mr.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_hh_b7_mr_flat_singleton_20260822_chtc"
CAMPAIGN_ID = "paper_i_hh_b7_mr_flat_singleton_20260822"
PLAN_SCHEMA = "b7mrfs_plan_v1"
AUTH_SCHEMA = "b7mrfs_authorization_v1"
JOB_SCHEMA = "b7mrfs_job_v1"
MANIFEST_SCHEMA = "b7mrfs_package_v1"
WORKER_SCHEMA = "b7mrfs_worker_v1"
RESULT_MANIFEST_SCHEMA = "b7mrfs_result_v1"

SCHUR_APPEND_CANDIDATE_WARM_START_POLICY = (
    "schur_append_candidate_warm_start_v1"
)
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

MR_ALGORITHM_ID = (
    "paper_i_ra_adapt_nonstationary_incremental_full_response_v2"
)
MR_ACTIVE_GRADIENT_POLICY = "measured_residual_response_v1"
MR_CANDIDATE_GAIN_POLICY = "joint_minus_active_only_supported_trust_v1"

EXPLICIT_DEPTH_COST = "explicit_depth_qiskit_cost_v1"
COSTEXP_EQUIVALENCE = (
    "termwise_cross_component_commutation_min_cost_realization_v1"
)

# (regime_id, nph, horizon): weak sector -> 50, strong sector -> 75.
REGIMES = (
    ("weak_weak", 3, 50),
    ("intermediate_weak", 3, 50),
    ("strong_weak_u8", 3, 50),
    ("weak_strong", 7, 75),
    ("intermediate_strong", 7, 75),
    ("strong_strong_u8", 7, 75),
)

_POSITION_BUILDER = (
    "build_paper_i_ra_all_phase_position_adaptive_forced_k50_request"
)
_POSITION_ROUTE = (
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_FORCED_K50_V1"
)
_APPEND_BUILDER = (
    "build_paper_i_ra_all_phase_adaptive_forced_k50_request"
)
_APPEND_ROUTE = (
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_K50_V1"
)

ARMS = (
    {
        "arm_id": "mr_append_ra",
        "insertion_policy": "append_only",
        "insertion_class": "AppendOnlyInsertion",
    },
    {
        "arm_id": "mr_plateau_insertion",
        "insertion_policy": "plateau_commutation",
        "insertion_class": "PlateauCommutationInsertion",
    },
    {
        "arm_id": "mr_always_insertion",
        "insertion_policy": "always_commutation_reduced",
        "insertion_class": "AlwaysCommutationReducedInsertion",
    },
)

EXPECTED_DEPTH_WEIGHTS = {
    "weight_2q": 0.3,
    "weight_depth_2q": 0.3,
    "weight_depth": 0.25,
    "weight_1q": 0.0,
    "weight_size": 0.0,
}

EXPECTED_COSTEXP_BASELINE_PENALTY_WEIGHTS = {
    "penalty_weight_2q": 1.0,
    "penalty_weight_depth_2q": 0.0,
    "penalty_weight_depth": 0.1,
    "penalty_weight_1q": 0.0,
    "penalty_weight_size": 0.01,
}

EXPECTED_DEPTH_PENALTY_WEIGHTS = {
    "penalty_weight_2q": 0.3,
    "penalty_weight_depth_2q": 0.3,
    "penalty_weight_depth": 0.25,
    "penalty_weight_1q": 0.0,
    "penalty_weight_size": 0.0,
}

# Strong cells inherit the proven 40960 envelope (ladder to 65536/81920
# on OOM); weak nph3 cells are far lighter.
RESOURCE_ENVELOPES = {
    3: {
        "cpus": 4,
        "memory_mb": 16384,
        "disk_mb": 30720,
        "runtime_seconds": 172800,
    },
    7: {
        "cpus": 4,
        "memory_mb": 40960,
        "disk_mb": 61440,
        "runtime_seconds": 259200,
    },
}


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "sha256" in result:
        raise ValueError("payload already contains sha256")
    result["sha256"] = canonical_sha256(result)
    return result


def verify_digested(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    result = dict(payload)
    observed = result.pop("sha256", None)
    if result.get("schema") != schema or observed != canonical_sha256(result):
        raise ValueError(f"{schema} receipt drifted")
    result["sha256"] = observed
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON mapping")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_id(arm_id: str, regime_id: str, nph: int) -> str:
    return f"b7mr__{arm_id}__{regime_id}__nph{nph}"


def expected_cells() -> list[tuple[Mapping[str, Any], str, int, int]]:
    """(arm, regime_id, nph, horizon) for the 18 fresh cells."""
    return [
        (arm, regime_id, nph, horizon)
        for arm in ARMS
        for regime_id, nph, horizon in REGIMES
    ]
