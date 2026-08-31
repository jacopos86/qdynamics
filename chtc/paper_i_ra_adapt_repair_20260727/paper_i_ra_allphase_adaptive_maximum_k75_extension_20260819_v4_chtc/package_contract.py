#!/usr/bin/env python3
"""Closed contract for the k75 extension RECOVERY package (v2).

v1 (cluster 9663770) sealed a source whose reporting layer enumerated
canonical RA identities only for horizons 1..50; every cell therefore
crashes inside summarize_paper_i_run immediately after finishing round
75. v2 reseals the corrected source and finishes the campaign in two
job kinds:

- ``resume_live`` (6 cells: plateau_position and always_open, all three
  strong regimes): same-horizon AcceptedStateResume from each v1 job's
  freshest pulled mid-flight checkpoint trio. Loses at most the rounds
  since the pull.
- ``completion`` (3 append cells): zero-recompute completion. The v1
  jobs' round-75 checkpoint pairs (recovered from their crash transfer
  tarballs) hydrate under the corrected source with an operational cap
  of horizon+1; the protocol stop fires before any new round, and the
  summary/result artifacts are written properly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKAGE_ID = "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4_chtc"
CAMPAIGN_ID = "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819_v4"
PLAN_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_plan_v4"
AUTH_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_authorization_v4"
JOB_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_job_v4"
MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_package_v4"
WORKER_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_worker_v4"
RESULT_MANIFEST_SCHEMA = "paper_i_ra_allphase_adaptive_maximum_k75_result_v4"
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
PULLED_CHECKPOINTS = (
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/"
    "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819/live_tracking/checkpoints"
)
CRASH_TARBALLS = (
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/local_runs/"
    "paper_i_ra_allphase_adaptive_maximum_k75_extension_20260819/v1_crash_tarballs"
)
# (arm_id, regime) -> mode; append cells complete from their v1 round-75
# crash checkpoints, the other six resume their pulled mid-flight state.
CELL_PLAN: dict[tuple[str, str], tuple[str, str | None]] = {
    (str(arm["arm_id"]), regime): (
        "deferred" if arm["arm_id"] == "append_ra" else "resume_live",
        "append completion needs a reviewed zero-recompute mechanism; "
        "round-75 crash checkpoints are banked"
        if arm["arm_id"] == "append_ra"
        else None,
    )
    for arm in ARMS
    for regime, _nph in REGIMES
}
# All three always-open v1 cells restart fresh: intermediate_strong
# OOM-held with no output transfer (sandbox gone), and the weak_strong /
# strong_strong_u8 pulled bundles are cross-epoch pairs (ledger sidecar
# not referenced by current.json — the write-order hazard in
# HANDOFF_REFACTOR_cost_path_20260819.md item 6). The plateau trio's
# bundles passed full pair-coherence checks (state, ledger, resume
# sidecar) and resume. v3 source includes the fork-pool scoring stage
# (parity: byte-identical) — fresh restarts recompute on 4 cores.
# v4: all six cells resume their own v3 checkpoints (coherence-safe
# pulls) under the byte-clean fork-pool source — the "continue under
# parallelized code" step of the user's plan. Nothing is fresh.

RESOURCE_ENVELOPES = {
    # 24576 proven insufficient at resumed depths (three v3 plateau cells
    # OOM-held); start at the 40G rung.
    7: {"cpus": 4, "memory_mb": 40960, "disk_mb": 61440, "runtime_seconds": 259200},
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
    """(arm, regime_id, nph, mode, source) for the 9 recovery cells."""
    rows = []
    for arm in ARMS:
        for regime_id, nph in REGIMES:
            mode, source = CELL_PLAN[(str(arm["arm_id"]), regime_id)]
            if mode == "deferred":
                continue
            rows.append((arm, regime_id, nph, mode, source))
    return rows
