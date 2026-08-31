#!/usr/bin/env python3
"""Closed contract for the six-regime costexp + depth-fix package (v1).

User program 2026-08-20 (matrices A and B; matrix C with the alpha-star
warm start ships separately once its axis is sealed):

- Matrix A (12 cells): the two cost-expanded insertion kinds alone —
  always-open and plateau-position — across ALL SIX canonical regimes.
  Baseline signed routed penalty; no explicit-depth overlay.
- Matrix B (18 cells): the three RA routes across all six regimes with
  the explicit three-term Qiskit cost
  (K = 0.30 dN2q + 0.30 dD2q + 0.25 dD; 1q and size retired at zero)
  AND the commutation-reduced min-cost position realization where
  position classes exist (append has endpoint-only insertion, so it
  carries the cost term without position classes).

Horizons: weak-Holstein sector (nph=3) runs to 50; strong-Holstein
sector (nph=7) runs to 75. Fresh starts from iteration 0.

Not a Paper-I matrix; does not select the retired SNAKE route. Legacy
``paper_i``/``sr_snake`` names below are source namespaces only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "ra_adapt_costexp_depthfix_sixregime_20260820_v1_chtc"
CAMPAIGN_ID = "ra_adapt_costexp_depthfix_sixregime_20260820_v1"
PLAN_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_plan_v1"
AUTH_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_authorization_v1"
JOB_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_job_v1"
MANIFEST_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_package_v1"
WORKER_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_worker_v1"
RESULT_MANIFEST_SCHEMA = "ra_adapt_costexp_depthfix_sixregime_result_v1"
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

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
    "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request"
)
_POSITION_ROUTE = (
    "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2"
)
_APPEND_BUILDER = (
    "build_paper_i_ra_all_phase_adaptive_natural_terminal_request"
)
_APPEND_ROUTE = (
    "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2"
)

ARMS = (
    # Matrix A: costexp insertion alone
    {
        "arm_id": "a_costexp_always_open",
        "matrix": "A",
        "family": "costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "always_commutation_reduced_costexp",
        "compile_cost_policy": None,
    },
    {
        "arm_id": "a_costexp_plateau_position",
        "matrix": "A",
        "family": "costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "plateau_commutation_costexp",
        "compile_cost_policy": None,
    },
    # Matrix B: three routes x explicit-depth cost, with the position
    # fix on the two insertion routes
    {
        "arm_id": "b_depth_append_ra",
        "matrix": "B",
        "family": "explicit_depth",
        "builder": _APPEND_BUILDER,
        "route_constant": _APPEND_ROUTE,
        "insertion_policy": "append_only",
        "compile_cost_policy": EXPLICIT_DEPTH_COST,
    },
    {
        "arm_id": "b_depth_costexp_always_open",
        "matrix": "B",
        "family": "explicit_depth_costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "always_commutation_reduced_costexp",
        "compile_cost_policy": EXPLICIT_DEPTH_COST,
    },
    {
        "arm_id": "b_depth_costexp_plateau_position",
        "matrix": "B",
        "family": "explicit_depth_costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "plateau_commutation_costexp",
        "compile_cost_policy": EXPLICIT_DEPTH_COST,
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
    return f"sixregime__{arm_id}__{regime_id}__nph{nph}"


def expected_cells() -> list[tuple[Mapping[str, Any], str, int, int]]:
    """(arm, regime_id, nph, horizon) for the 30 fresh cells."""
    return [
        (arm, regime_id, nph, horizon)
        for arm in ARMS
        for regime_id, nph, horizon in REGIMES
    ]
