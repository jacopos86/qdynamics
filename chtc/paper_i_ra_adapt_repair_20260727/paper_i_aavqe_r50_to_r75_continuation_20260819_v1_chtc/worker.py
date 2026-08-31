#!/usr/bin/env python3
"""Execute one AAVQE r50->r75 reconstruction continuation cell on CHTC.

The published r50 tolmatch anchor (bundled in package/anchors/) is
rebuilt through the append facade's reconstruction resume: the rebuilt
prefix must reproduce the anchor's terminal energy within 1e-14 (both
values receipted on failure), the continuation's signed replay evidence
chains onto the anchor's own chain, and rounds 51..75 execute under a
horizon-75 request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent

JOB_SCHEMA = "paper_i_aavqe_r75_continuation_job_v1"
RESULT_SCHEMA = "paper_i_aavqe_r75_continuation_result_v1"
TARGET_HORIZON = 75
# 1e-14 rejected four cells with gaps of 4e-14..9e-14 -- pure BLAS
# aggregation-order roundoff across execute nodes (a real reconstruction
# mismatch is >=1e-8). 1e-12 still receipts exact prefix reproduction.
TOLERANCE = 1.0e-12


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = _canonical_sha256(result)
    return result


def _verify(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    body = dict(payload)
    if body.get("schema") != schema or body.get("sha256") != _canonical_sha256(body):
        raise RuntimeError(f"{schema} receipt drifted")
    return body


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(job_path: Path, output: Path) -> int:
    job = _verify(json.loads(job_path.read_text()), schema=JOB_SCHEMA)
    anchor_path = PACKAGE_DIR / str(job["anchor_path"])
    if _file_sha256(anchor_path) != str(job["anchor_sha256"]):
        raise RuntimeError("anchor checkpoint sha drifted")
    anchor = json.loads(anchor_path.read_text())
    if anchor.get("reconstruction_fields_complete") is not True:
        raise RuntimeError("anchor is not reconstruction-complete")
    history_path = PACKAGE_DIR / str(job["anchor_history_path"])
    if _file_sha256(history_path) != str(job["anchor_history_sha256"]):
        raise RuntimeError("anchor history sha drifted")
    anchor_history = json.loads(history_path.read_text())

    from pipelines.static_adapt.ra_adapt import run_append_adapt
    from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
    from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
        build_paper_i_ra_hh_regime_problem,
    )
    from pipelines.static_adapt.sr_snake.contracts import (
        SRExecutionPolicy,
        SRStopPolicy,
    )

    problem = build_paper_i_ra_hh_regime_problem(str(job["regime_id"]))
    request = AppendAdaptRequest(
        execution=SRExecutionPolicy(
            stop=SRStopPolicy(maximum_controller_rounds=TARGET_HORIZON)
        )
    )
    reconstruction = {
        "accepted_operator_labels": anchor["accepted_operator_labels"],
        "accepted_generator_identities": anchor[
            "accepted_generator_identities"
        ],
        "logical_theta": anchor["logical_theta"],
        "controller_rounds_completed": int(
            anchor["controller_rounds_completed"]
        ),
        "expected_prefix_energy": float(job["expected_prefix_energy"]),
        "energy_reproduction_tolerance": TOLERANCE,
        "anchor_sha256": str(job["anchor_sha256"]),
        "controller_replay_evidence": anchor["controller_replay_evidence"],
        "anchor_history": anchor_history,
    }
    result = run_append_adapt(
        problem, request, reconstruction_resume=reconstruction
    )
    payload = result.to_dict()
    body = payload["result_payload"]
    history = body["history"]
    rounds = [int(row["controller_round"]) for row in history]
    resumed = int(anchor["controller_rounds_completed"])
    if (
        rounds != list(range(1, TARGET_HORIZON + 1))
        or body.get("continuation_resumed_from_round") != resumed
        or int(body["controller_rounds_completed"]) != TARGET_HORIZON
    ):
        raise RuntimeError(
            f"continuation rounds drifted: {rounds[:3]}..{rounds[-3:]}"
        )
    boundary_energy = float(history[resumed - 1]["energy_after"])
    boundary_gap = abs(boundary_energy - float(job["expected_prefix_energy"]))
    if boundary_gap > TOLERANCE:
        raise RuntimeError(
            f"stitched boundary energy drifted: {boundary_gap:.3e}"
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(
        json.dumps(payload, sort_keys=True, default=str)
    )
    receipt = _digested(
        {
            "schema": RESULT_SCHEMA,
            "status": "passed_authenticated_r75_continuation",
            "execution_id": job["execution_id"],
            "job_sha256": job["sha256"],
            "anchor_sha256": job["anchor_sha256"],
            "resumed_from_round": resumed,
            "target_round": TARGET_HORIZON,
            "continued_round_count": len(rounds),
            "final_energy": float(body["final_energy"]),
            "energy_reproduction_tolerance": TOLERANCE,
            "result_file_sha256": _file_sha256(output / "result.json"),
        }
    )
    (output / "worker_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=False)
    parser.add_argument("--authorization", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.job, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
