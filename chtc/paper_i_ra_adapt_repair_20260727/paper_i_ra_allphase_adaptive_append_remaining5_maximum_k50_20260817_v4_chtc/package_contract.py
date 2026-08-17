#!/usr/bin/env python3
"""Closed contract for the three-arm all-phase-adaptive CHTC matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_20260817_v4_chtc"
CAMPAIGN_ID = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_20260817_v4"
PLAN_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_plan_v1"
AUTH_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_authorization_v1"
JOB_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_job_v1"
MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_package_v1"
WORKER_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_worker_v1"
RESULT_MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_append_remaining5_maximum_k50_result_v1"
TARGET_HORIZON = 50
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

REGIMES = (
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ARMS = (
    {
        "arm_id": "append_position_phase0",
        "builder": "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request",
        "route_constant": "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
        "insertion_policy": "append_only",
        "phase0_population": "insertion_position_records",
        "phase123_population": "insertion_position_records",
    },
)
RESOURCE_ENVELOPES = {
    3: {"cpus": 4, "memory_mb": 8192, "disk_mb": 20480, "runtime_seconds": 259200},
    7: {"cpus": 4, "memory_mb": 12288, "disk_mb": 30720, "runtime_seconds": 259200},
}


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


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
    return f"allphase_maxk50__{arm_id}__{regime_id}__nph{nph}"


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(
        execution_id(str(arm["arm_id"]), regime, nph)
        for arm in ARMS
        for regime, nph in REGIMES
    )

