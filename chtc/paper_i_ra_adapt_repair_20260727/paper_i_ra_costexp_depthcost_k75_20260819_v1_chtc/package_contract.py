#!/usr/bin/env python3
"""Closed contract for the costexp + explicit-depth-cost k75 package (v1).

Fifteen fresh h75 cells over the three strong-Holstein regimes:

- costexp arms (2): always-open and plateau-position with the
  cost-expanded insertion kinds — the commutation-reduced gradient is
  scored once per response class while the routed compile cost is
  evaluated at every member position and the minimum-penalty realization
  is admitted (earliest member on exact ties).  No depth term.
- explicit-depth arms (3): append, always-open, and plateau-position
  under ``explicit_depth_qiskit_cost_v1`` — the exact FakeMarrakesh
  before-minus-after marginals reweighted as 0.30 two-qubit count,
  0.30 two-qubit depth, 0.25 full compiled circuit depth; one-qubit
  basis work and size retired at zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_ra_costexp_depthcost_k75_20260819_v1_chtc"
CAMPAIGN_ID = "paper_i_ra_costexp_depthcost_k75_20260819_v1"
PLAN_SCHEMA = "paper_i_ra_costexp_depthcost_k75_plan_v1"
AUTH_SCHEMA = "paper_i_ra_costexp_depthcost_k75_authorization_v1"
JOB_SCHEMA = "paper_i_ra_costexp_depthcost_k75_job_v1"
MANIFEST_SCHEMA = "paper_i_ra_costexp_depthcost_k75_package_v1"
WORKER_SCHEMA = "paper_i_ra_costexp_depthcost_k75_worker_v1"
RESULT_MANIFEST_SCHEMA = "paper_i_ra_costexp_depthcost_k75_result_v1"
TARGET_HORIZON = 75
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

EXPLICIT_DEPTH_COST = "explicit_depth_qiskit_cost_v1"
COSTEXP_EQUIVALENCE = (
    "termwise_cross_component_commutation_min_cost_realization_v1"
)

REGIMES = (
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
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
    {
        "arm_id": "costexp_always_open",
        "family": "costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "always_commutation_reduced_costexp",
        "compile_cost_policy": None,
    },
    {
        "arm_id": "costexp_plateau_position",
        "family": "costexp",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "plateau_commutation_costexp",
        "compile_cost_policy": None,
    },
    {
        "arm_id": "depth_append_ra",
        "family": "explicit_depth",
        "builder": _APPEND_BUILDER,
        "route_constant": _APPEND_ROUTE,
        "insertion_policy": "append_only",
        "compile_cost_policy": EXPLICIT_DEPTH_COST,
    },
    {
        "arm_id": "depth_always_open",
        "family": "explicit_depth",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "always_commutation_reduced",
        "compile_cost_policy": EXPLICIT_DEPTH_COST,
    },
    {
        "arm_id": "depth_plateau_position",
        "family": "explicit_depth",
        "builder": _POSITION_BUILDER,
        "route_constant": _POSITION_ROUTE,
        "insertion_policy": "plateau_commutation",
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

# 40960 MB from the v4 ladder lesson: plateau cells OOM-held at 24576 with
# no output transfer; fresh runs reach the same late-round footprint.
RESOURCE_ENVELOPES = {
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
    return f"costexp_depth_k75__{arm_id}__{regime_id}__nph{nph}"


def expected_cells() -> list[tuple[Mapping[str, Any], str, int]]:
    """(arm, regime_id, nph) for the 15 fresh cells."""
    return [
        (arm, regime_id, nph)
        for arm in ARMS
        for regime_id, nph in REGIMES
    ]
