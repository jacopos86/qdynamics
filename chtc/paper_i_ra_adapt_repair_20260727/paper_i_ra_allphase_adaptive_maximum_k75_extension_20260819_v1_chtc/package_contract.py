#!/usr/bin/env python3
"""Closed contract for the maximum-family k75 strong-sector extension.

User authorization (2026-08-18, in chat): continue the always-insertion
campaign strong-sector cells and the plateau-insertion cells to
iteration 75 on CHTC. All twelve cells (4 maximum-family arms x 3 strong
regimes) run FRESH under horizon-75 protocols: determinism reproduces
each cell's k50 prefix exactly (smoke A evidence), the k50 grid
completing in parallel supplies the matched-horizon dataset, and the
rounds-1..50 agreement between the two lanes is the behavior-preservation
cross-check. See CELL_PLAN comment for why resume-based continuation was
set aside (uniform per-round route identity in the provenance surface).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v1_chtc"
CAMPAIGN_ID = "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v1"
PLAN_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_plan_v1"
AUTH_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_authorization_v1"
JOB_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_job_v1"
MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_package_v1"
WORKER_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_worker_v1"
RESULT_MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_result_v1"
TARGET_HORIZON = 75
RESUME_ROUND = 50
IMAGE_PATH = "chtc/phase3_optuna/image.sif"
IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"

REGIMES = (
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)
ARMS = (
    {
        "arm_id": "append_ra",
        "builder": "build_paper_i_ra_all_phase_adaptive_natural_terminal_request",
        "route_constant": "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
        "insertion_policy": "append_only",
        "phase0_population": "append_endpoint_generators",
        "phase123_population": "append_records",
    },
    {
        "arm_id": "always_open_position_phase0",
        "builder": "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request",
        "route_constant": "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
        "insertion_policy": "always_commutation_reduced",
        "phase0_population": "insertion_position_records",
        "phase123_population": "insertion_position_records",
    },
    {
        "arm_id": "plateau_append_phase0",
        "builder": "build_paper_i_ra_all_phase_adaptive_natural_terminal_request",
        "route_constant": "PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
        "insertion_policy": "plateau_commutation",
        "phase0_population": "append_endpoint_generators",
        "phase123_population": "insertion_position_records",
    },
    {
        "arm_id": "plateau_position_phase0",
        "builder": "build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request",
        "route_constant": "PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2",
        "insertion_policy": "plateau_commutation",
        "phase0_population": "insertion_position_records",
        "phase123_population": "insertion_position_records",
    },
)

_K50_ARCHIVES = (
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/"
    "paper_i_ra_allphase_adaptive_20260817_comparison_data/archives"
)
# All twelve strong-sector cells run FRESH at horizon 75. Decision
# 2026-08-19 03:0x: the resume-based continuation authenticates in the
# loader (horizon-extension rule, committed) but the summary layer and
# every downstream consumer enforce uniform per-round route identity by
# design; rather than weaken that provenance surface, fresh h75 runs are
# used — determinism reproduces the k50 prefix exactly (smoke A: operator
# sequence and positions identical, energies <= 2.2e-15), and the k50
# grid completing in parallel provides the matched-horizon dataset plus
# the rounds-1..50 cross-check for every h75 run.
CELL_PLAN: dict[tuple[str, str], tuple[str, str | None]] = {
    (str(arm["arm_id"]), regime): ("fresh", None)
    for arm in ARMS
    for regime, _nph in REGIMES
}

RESOURCE_ENVELOPES = {
    7: {"cpus": 4, "memory_mb": 24576, "disk_mb": 61440, "runtime_seconds": 259200},
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
    return f"allphase_k75__{arm_id}__{regime_id}__nph{nph}"


def expected_cells() -> list[tuple[Mapping[str, Any], str, int, str, str | None]]:
    """(arm, regime_id, nph, mode, resume_archive) for the 12 fresh cells."""
    rows = []
    for arm in ARMS:
        for regime_id, nph in REGIMES:
            mode, source = CELL_PLAN[(str(arm["arm_id"]), regime_id)]
            if mode == "deferred":
                continue
            rows.append((arm, regime_id, nph, mode, source))
    return rows
