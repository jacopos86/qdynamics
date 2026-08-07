"""Run the first lossless-bilinear hidden-order gate for the Holstein dimer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from paper5.stability import DimerParameters
from paper5.stability.reachability_observability import (
    drive_aware_word_hankel_rank_audit,
)

RUN_ID = "paper_v_lossless_bilinear_word_hankel_gate_20260804_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(
    output_directory: Path,
    *,
    cutoffs: tuple[int, ...] = (12, 16),
    rank_tolerances: tuple[float, ...] = (1e-10, 1e-12),
    maximum_word_depth: int = 3,
    practical_hidden_budget: int = 96,
) -> dict[str, object]:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
        pulse_width=1.0,
    )
    plan: dict[str, object] = {
        "schema": "paper_v_lossless_bilinear_word_hankel_plan_v1",
        "run_id": RUN_ID,
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "execution_authorized": True,
        "scientific_question": (
            "Does the drive-aware unresolved force channel alone impose a "
            "hidden-order lower bound above the practical r=96 budget?"
        ),
        "parameters": {
            "hopping": parameters.hopping,
            "omega_ph": parameters.omega_ph,
            "coupling": parameters.coupling,
            "lambda_ep": parameters.lambda_ep,
            "gamma": parameters.gamma,
        },
        "audit": {
            "cutoffs": list(cutoffs),
            "rank_tolerances": list(rank_tolerances),
            "maximum_word_depth": maximum_word_depth,
            "practical_hidden_budget": practical_hidden_budget,
            "component_words": ["static", "drive"],
            "preparation_columns": (
                "not required if the force-channel lower bound already exceeds "
                "the budget; adding them cannot reduce Hankel rank"
            ),
            "exact_trajectory_access": "none",
            "autonomous_rollout": "forbidden before this gate passes",
        },
    }
    _write_json_atomic(output_directory / "plan.json", plan)

    started = time.time()
    audits = []
    for cutoff in cutoffs:
        for tolerance in rank_tolerances:
            print(
                f"cutoff={cutoff} rank_tolerance={tolerance:.1e}",
                flush=True,
            )
            audit = drive_aware_word_hankel_rank_audit(
                parameters,
                phonon_cutoff=cutoff,
                maximum_word_depth=maximum_word_depth,
                rank_tolerance=tolerance,
                practical_hidden_budget=practical_hidden_budget,
            )
            print(
                "  new ranks="
                f"{audit.new_ranks}; cumulative={audit.cumulative_ranks}",
                flush=True,
            )
            audits.append(audit)

    grouped: dict[int, list] = {}
    for audit in audits:
        grouped.setdefault(audit.phonon_cutoff, []).append(audit)
    tolerance_stable = all(
        max(row.hankel_rank_lower_bound for row in rows)
        - min(row.hankel_rank_lower_bound for row in rows)
        <= 2
        for rows in grouped.values()
    )
    all_crossed = all(audit.crossed_budget for audit in audits)
    gate_passed = tolerance_stable and not all_crossed
    if all_crossed and tolerance_stable:
        decision = "stop_archive_hidden_realization_budget_exceeded"
    elif not tolerance_stable:
        decision = "indeterminate_word_rank_not_tolerance_stable"
    else:
        decision = "continue_to_preparation_and_time_limited_hankel_audit"

    summary: dict[str, object] = {
        "schema": "paper_v_lossless_bilinear_word_hankel_summary_v1",
        "run_id": RUN_ID,
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "status": "complete",
        "gate_passed": gate_passed,
        "decision": decision,
        "tolerance_stable_within_two": tolerance_stable,
        "all_force_channel_lower_bounds_exceed_budget": all_crossed,
        "practical_hidden_budget": practical_hidden_budget,
        "audits": [audit.summary() for audit in audits],
        "preparation_audit_executed": False,
        "time_limited_audit_executed": False,
        "omission_reason": (
            "The force-channel word-Hankel lower bound already exceeds the "
            "practical hidden-state budget. Preparation columns can only leave "
            "that rank unchanged or increase it."
            if all_crossed
            else "The next audit remains required."
        ),
        "exact_trajectory_used": False,
        "autonomous_rollout_executed": False,
        "elapsed_seconds": time.time() - started,
    }
    _write_json_atomic(output_directory / "summary.json", summary)

    repo_root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(__file__).resolve(),
        repo_root
        / "paper_5/src/paper5/stability/reachability_observability.py",
        repo_root / "paper_5/src/paper5/stability/krylov_memory_closure.py",
        repo_root / "paper_5/src/paper5/stability/exact_reference.py",
    )
    artifact_paths = (
        output_directory / "plan.json",
        output_directory / "summary.json",
    )
    manifest: dict[str, object] = {
        "schema": "paper_v_lossless_bilinear_word_hankel_manifest_v1",
        "run_id": RUN_ID,
        "status": "complete",
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "python": sys.version,
        "platform": platform.platform(),
        "source_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path) for path in artifact_paths
        },
    }
    _write_json_atomic(output_directory / "runtime_manifest.json", manifest)
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/local_runs") / RUN_ID,
    )
    parser.add_argument("--cutoffs", type=_parse_csv_ints, default=(12, 16))
    parser.add_argument(
        "--rank-tolerances",
        type=_parse_csv_floats,
        default=(1e-10, 1e-12),
    )
    parser.add_argument("--maximum-word-depth", type=int, default=3)
    parser.add_argument("--practical-hidden-budget", type=int, default=96)
    args = parser.parse_args()
    summary = run(
        args.output_directory,
        cutoffs=args.cutoffs,
        rank_tolerances=args.rank_tolerances,
        maximum_word_depth=args.maximum_word_depth,
        practical_hidden_budget=args.practical_hidden_budget,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
