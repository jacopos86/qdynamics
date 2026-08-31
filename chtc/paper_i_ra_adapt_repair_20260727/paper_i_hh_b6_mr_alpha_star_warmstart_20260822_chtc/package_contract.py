#!/usr/bin/env python3
"""Closed contract for B6 (matrix C): B5 plus alpha-star Schur candidate warm start.

B6 is the companion the B3-MR plan promised: identical to B5 (b3mr, cluster
9674089) in every axis except one — the request's method carries
``candidate_initialization = SchurAppendCandidateWarmStart()``, activating
``adapt_schur_warm_start_mode="append"``: each phase-3 trial refit seeds ONLY
the inserted or appended candidate's coordinate from the guarded Schur window
solve (signed proposals, evaluate-both). Accepted response parameters are never
warm-started; the joint-step response warm start rides the route unchanged from
B5. The sealed axis landed on ra-refactor-stage2 @ 5775f0eb, which is also the
source freeze for this package (unlike b3mr's earlier freeze, which predates
the axis).

Any B6 vs B5 table is a one-axis comparison and must say so. Everything below
the campaign identifiers is deliberately byte-identical to the b3mr contract.

--- b3mr contract preserved for lineage ---

B3 ran the semantic-closure selector (stationary source response, raw joint
gain). B3-MR runs the canonical non-stationary lane: the active block's
residual gradient is MEASURED every round and a candidate is credited only
with its marginal gain (joint minus active-only). These are one coupled
commitment, not two knobs.

B3 ran the semantic-closure selector (stationary source response, raw joint
gain). B3-MR runs the canonical non-stationary lane: the active block's
residual gradient is MEASURED every round and a candidate is credited only
with its marginal gain (joint minus active-only). These are one coupled
commitment, not two knobs.

Route (verified at runtime, 2026-08-21 smoke, weak_weak 8 rounds):
  algorithm_id            = paper_i_ra_adapt_nonstationary_incremental_full_response_v2
  active_gradient_policy  = measured_residual_response_v1
  phase3_candidate_gain   = joint_minus_active_only_supported_trust_v1
N_grad 1275 of S_alg 4041 -> the gradient is genuinely acquired, not assumed.

DISCLOSURE - B3-MR is NOT a one-axis change from B3. Two further differences
are structural to this lane and were proven by runtime smoke, not chosen:
  1. cost term: costexp insertions abort with "Cost-expanded Qiskit member
     lost a routed penalty input"; plain insertion + explicit-depth cost
     aborts with "Explicit-depth compile-cost route is not bound to its
     sealed weight set". B3-MR therefore selects with NO compile-cost term,
     where B3 used K = 0.30 dN2q + 0.30 dD2q + 0.25 dD.
  2. insertion positions: B3's insertion arms used min-cost realization
     within the commutation class; B3-MR uses plain commutation insertion.
Also rides the route (per handoff): whitened chart + exact-guarded joint-step
warm start. Any B3-MR vs B3 table must name all of these.

Matrix: 3 insertion policies x 6 regimes = 18 cells. Same pools, seeds
(adapt 7, transpiler 7), cutoffs (weak nph3, strong nph7) and horizons as B3
(weak -> 50, strong -> 75). r45 is a REPORTING convention, not a stop rule.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_hh_b6_mr_alpha_star_warmstart_20260822_chtc"
CAMPAIGN_ID = "paper_i_hh_b6_mr_alpha_star_warmstart_20260822"
PLAN_SCHEMA = "b6mrws_plan_v1"
AUTH_SCHEMA = "b6mrws_authorization_v1"
JOB_SCHEMA = "b6mrws_job_v1"
MANIFEST_SCHEMA = "b6mrws_package_v1"
WORKER_SCHEMA = "b6mrws_worker_v1"
RESULT_MANIFEST_SCHEMA = "b6mrws_result_v1"

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
    return f"b6mr__{arm_id}__{regime_id}__nph{nph}"


def expected_cells() -> list[tuple[Mapping[str, Any], str, int, int]]:
    """(arm, regime_id, nph, horizon) for the 18 fresh cells."""
    return [
        (arm, regime_id, nph, horizon)
        for arm in ARMS
        for regime_id, nph, horizon in REGIMES
    ]
