#!/usr/bin/env python3
"""Run one fresh, source-locked Paper-I Hubbard L=4 arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.ra_adapt.append import run_append_adapt
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.static_adapt.ra_adapt.paper_i_hubbard_l4 import (
    PAPER_I_HUBBARD_L4_EXACT_ENERGY,
    build_paper_i_hubbard_l4_append_request,
    build_paper_i_hubbard_l4_problem,
    build_paper_i_hubbard_l4_request,
    materialize_paper_i_hubbard_l4_append_protocol,
    materialize_paper_i_hubbard_l4_ra_protocol,
    paper_i_hubbard_l4_application_source_contract,
)


ARMS = ("append", "baseline", "prune", "batch", "beam")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode(
            "utf-8"
        )
        + b"\n"
    )
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _compact_summary(
    *,
    arm: str,
    requested_rounds: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if arm == "append":
        reporting = result["paper_i_summary"]
        trajectory = list(reporting["accepted_history"])
        error_trace = [
            {
                "controller_round": int(row["controller_round"]),
                "active_ansatz_depth": int(row["controller_round"]),
                "accepted_energy": float(row["energy_after"]),
                "exact_same_cutoff_energy": (
                    PAPER_I_HUBBARD_L4_EXACT_ENERGY
                ),
                "absolute_energy_error": abs(
                    float(row["energy_after"])
                    - PAPER_I_HUBBARD_L4_EXACT_ENERGY
                ),
            }
            for row in trajectory
        ]
        accounting = reporting["estimator_accounting"]
        canonical_all_work = {
            "components": {
                "n_h_outer": int(accounting["N_H_outer"]),
                "n_h_refit": int(accounting["N_H_refit"]),
                "n_grad": int(accounting["N_grad"]),
                "n_metric": int(accounting["N_metric"]),
            },
            "s_alg": int(accounting["S_alg"]),
        }
        route_contract_sha256 = None
        prune_events: list[dict[str, Any]] = []
    else:
        run = result["run"]
        trajectory = list(run["accepted_trajectory"])
        reporting = run["paper_i_summary"]
        error_trace = list(reporting["accepted_error_trace"])
        canonical_all_work = reporting["canonical_all_work"]
        route_contract_sha256 = run["route"]["contract_sha256"]
        accepted_receipts = list(
            result["scientific_receipts"].get(
                "accepted_round_receipts",
                (),
            )
        )
        prune_events = [
            {
                "controller_round": int(row["controller_round"]),
                "receipt": row.get("prune"),
            }
            for row in accepted_receipts
            if row.get("prune") is not None
        ]
    terminal = error_trace[-1] if error_trace else None
    first_converged = next(
        (
            int(row["controller_round"])
            for row in error_trace
            if float(row["absolute_energy_error"]) <= 1.0e-8
        ),
        None,
    )
    return {
        "schema": "paper_i_hubbard_l4_arm_summary_v1",
        "arm": str(arm),
        "requested_rounds": int(requested_rounds),
        "accepted_controller_rounds": len(trajectory),
        "terminal": terminal,
        "first_absolute_error_le_1e-8": first_converged,
        "canonical_all_work": canonical_all_work,
        "prune_events": prune_events,
        "exact_same_sector_energy": PAPER_I_HUBBARD_L4_EXACT_ENERGY,
        "protocol_sha256": result["protocol"]["sha256"],
        "route_contract_sha256": route_contract_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    arm_dir = args.output_root.resolve() / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    problem = build_paper_i_hubbard_l4_problem()
    if args.arm == "append":
        request = build_paper_i_hubbard_l4_append_request(
            maximum_controller_rounds=args.rounds,
            output_dir=arm_dir,
        )
        protocol = materialize_paper_i_hubbard_l4_append_protocol(
            problem,
            request,
        )
        result_object = run_append_adapt(problem, protocol)
    else:
        request = build_paper_i_hubbard_l4_request(
            args.arm,
            maximum_controller_rounds=args.rounds,
            output_dir=arm_dir,
        )
        protocol = materialize_paper_i_hubbard_l4_ra_protocol(
            problem,
            request,
        )
        result_object = run_ra_adapt(problem, protocol)

    result = result_object.to_dict()
    if result["protocol"]["sha256"] != protocol.sha256:
        raise RuntimeError("Executed protocol does not match materialization.")
    source_contract = paper_i_hubbard_l4_application_source_contract(problem)
    summary = _compact_summary(
        arm=args.arm,
        requested_rounds=args.rounds,
        result=result,
    )
    result_sha256 = _sha256(result)
    summary_sha256 = _sha256(summary)
    diff = _git("diff", "--binary", "HEAD", "--")
    manifest = {
        "schema": "paper_i_hubbard_l4_arm_manifest_v1",
        "arm": args.arm,
        "requested_rounds": int(args.rounds),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain=v1"),
        "git_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "application_source_contract": source_contract,
        "protocol_sha256": protocol.sha256,
        "route_contract_sha256": summary["route_contract_sha256"],
        "result_sha256": result_sha256,
        "summary_sha256": summary_sha256,
        "optimizer": {
            "method": "Powell",
            "maxiter": 200,
            "xtol": 1.0e-5,
            "ftol": 1.0e-12,
        },
        "prune_nomination_gate": (
            {"parameter": "abs(theta_mu)", "threshold": 1.0e-10}
            if args.arm == "prune"
            else None
        ),
    }
    _atomic_json(arm_dir / "result.json", result)
    _atomic_json(arm_dir / "summary.json", summary)
    _atomic_json(arm_dir / "manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
