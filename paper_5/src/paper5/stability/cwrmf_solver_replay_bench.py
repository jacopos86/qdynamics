"""Run the offline replay bench over a recorded conic instance directory.

Usage:

    python -m paper5.stability.cwrmf_solver_replay_bench \\
        --instance-directory <run>/conic_instances \\
        --output-directory output/local_runs/paper_v_cwrmf_replay_... \\
        --candidates recorded,tolerance_1_decade,...

Every candidate is a declared setting set; ``--candidates all`` runs them in
the order they are declared below.  Results are written incrementally to
``bench.json`` so a killed run keeps what it measured.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cwrmf_solver_instances import load_instance_context, load_instances
from .cwrmf_solver_replay import (
    BALANCED_INFLATION_SHIFT,
    ClarabelCandidate,
    ReducedBundleCandidate,
    ToleranceCandidate,
    WarmStartedScsCandidate,
    score_candidate,
    summarize,
)


def declared_candidates(coefficients: Any) -> dict[str, Any]:
    """Return every named candidate, in the order the report tables them."""

    candidates: list[Any] = [
        ClarabelCandidate(
            "recorded",
            "the recorded Clarabel settings, unchanged",
            {},
        ),
        ToleranceCandidate(
            "tolerance_1_decade",
            "tol_gap_abs, tol_gap_rel and tol_feas multiplied by 10",
            1,
        ),
        ToleranceCandidate(
            "tolerance_2_decades",
            "tol_gap_abs, tol_gap_rel and tol_feas multiplied by 100",
            2,
        ),
        ClarabelCandidate(
            "direct_solve_method_qdldl",
            "direct_solve_method='qdldl' instead of 'auto'",
            {"direct_solve_method": "qdldl"},
        ),
        ClarabelCandidate(
            "direct_solve_method_faer",
            "direct_solve_method='faer' instead of 'auto'",
            {"direct_solve_method": "faer"},
        ),
        ClarabelCandidate(
            "equilibrate_off",
            "equilibrate_enable=False",
            {"equilibrate_enable": False},
        ),
        ClarabelCandidate(
            "equilibrate_max_iter_2",
            "equilibrate_max_iter=2 instead of 10",
            {"equilibrate_max_iter": 2},
        ),
        ClarabelCandidate(
            "presolve_off",
            "presolve_enable=False",
            {"presolve_enable": False},
        ),
        ClarabelCandidate(
            "static_regularization_off",
            "static_regularization_enable=False",
            {"static_regularization_enable": False},
        ),
        ClarabelCandidate(
            "static_regularization_1e_7",
            "static_regularization_constant=1e-7 instead of 1e-8",
            {"static_regularization_constant": 1e-7},
        ),
        ClarabelCandidate(
            "static_regularization_1e_10",
            "static_regularization_constant=1e-10 instead of 1e-8",
            {"static_regularization_constant": 1e-10},
        ),
        ClarabelCandidate(
            "chordal_decomposition_off",
            "chordal_decomposition_enable=False",
            {"chordal_decomposition_enable": False},
        ),
        ClarabelCandidate(
            "max_threads_2",
            "max_threads=2 instead of the pinned 1",
            {"max_threads": 2},
        ),
        ClarabelCandidate(
            "max_threads_4",
            "max_threads=4 instead of the pinned 1",
            {"max_threads": 4},
        ),
        ClarabelCandidate(
            "max_threads_8",
            "max_threads=8 instead of the pinned 1",
            {"max_threads": 8},
        ),
        WarmStartedScsCandidate(
            "scs_warm_1e_9",
            "SCS at eps_abs=eps_rel=1e-9, warm started from the previous "
            "instance's returned x",
            1e-9,
        ),
        WarmStartedScsCandidate(
            "scs_warm_1e_9_polish_strict",
            "the same warm-started SCS, followed by one Clarabel solve at "
            "the recorded settings whenever the strict acceptance test fails",
            1e-9,
            polish_profile="strict",
            coefficients=coefficients,
        ),
        WarmStartedScsCandidate(
            "scs_warm_1e_9_polish_balanced",
            "the same warm-started SCS, followed by one Clarabel solve at "
            "the recorded settings whenever the balanced acceptance test "
            "with the balanced profile's psd_inflation substituted for the "
            "strict value the operands carry fails",
            1e-9,
            polish_profile="balanced",
            coefficients=coefficients,
            polish_inflation_shift=BALANCED_INFLATION_SHIFT,
        ),
        ClarabelCandidate(
            "presolve_off_faer",
            "presolve_enable=False together with direct_solve_method='faer'",
            {"presolve_enable": False, "direct_solve_method": "faer"},
        ),
        ClarabelCandidate(
            "max_threads_8_presolve_off",
            "max_threads=8 together with presolve_enable=False",
            {"max_threads": 8, "presolve_enable": False},
        ),
        ClarabelCandidate(
            "recorded_repeat",
            "the recorded Clarabel settings again, to bracket the "
            "run-to-run spread of the machine over one bench pass",
            {},
        ),
        ReducedBundleCandidate(
            "reduced_bundle_1e_1",
            "keep only bundle directions whose compressed-base eigenvalue is "
            "at or below 1e-1 times the largest, then score the full 62-row "
            "Gram",
            1e-1,
        ),
        ReducedBundleCandidate(
            "reduced_bundle_1e_2",
            "keep only bundle directions whose compressed-base eigenvalue is "
            "at or below 1e-2 times the largest, then score the full 62-row "
            "Gram",
            1e-2,
        ),
        ReducedBundleCandidate(
            "reduced_bundle_1e_3",
            "keep only bundle directions whose compressed-base eigenvalue is "
            "at or below 1e-3 times the largest, then score the full 62-row "
            "Gram",
            1e-3,
        ),
        ReducedBundleCandidate(
            "reduced_bundle_1e_6",
            "keep only bundle directions whose compressed-base eigenvalue is "
            "at or below 1e-6 times the largest, then score the full 62-row "
            "Gram",
            1e-6,
        ),
    ]
    return {candidate.name: candidate for candidate in candidates}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        type=str,
        default="all",
        help="comma-separated candidate names, or 'all'",
    )
    parser.add_argument(
        "--instance-limit",
        type=int,
        default=None,
        help="score only the first N recorded instances (screening only)",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    coefficients, readable, metadata = load_instance_context(
        arguments.instance_directory
    )
    instances = load_instances(arguments.instance_directory)
    if arguments.instance_limit is not None:
        instances = instances[: arguments.instance_limit]
    catalogue = declared_candidates(coefficients)
    if arguments.candidates == "all":
        names = list(catalogue)
    else:
        names = [name.strip() for name in arguments.candidates.split(",")]
    report: dict[str, Any] = {
        "instance_directory": str(
            arguments.instance_directory.resolve()
        ),
        "instance_context": metadata,
        "instances": len(instances),
        "readable_completion_rates": int(readable.size),
        "candidates": {},
    }
    path = output_directory / "bench.json"
    for name in names:
        candidate = catalogue[name]
        print(f"[bench] {name}", flush=True)
        try:
            scores = score_candidate(candidate, instances, coefficients)
        except Exception as error:  # declared: a candidate may not apply
            report["candidates"][name] = {
                "description": candidate.description,
                "failed": f"{type(error).__name__}: {error}",
            }
            path.write_text(
                json.dumps(report, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[bench] {name} failed: {error}", flush=True)
            continue
        summary = summarize(scores)
        report["candidates"][name] = {
            "description": candidate.description,
            "summary": summary,
            "scores": [asdict(score) for score in scores],
        }
        path.write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[bench] {name} total={summary['total_wall_seconds']:.3f}s "
            f"strict_accept={summary['strict_accepted']}"
            f"(recorded {summary['recorded_strict_accepted']}) "
            f"strict_agree={summary['strict_verdict_agreement']}"
            f"/{summary['instances']} "
            f"balanced_shift_accept={summary['balanced_shifted_accepted']}"
            f"(recorded {summary['recorded_balanced_shifted_accepted']}) "
            f"balanced_shift_agree="
            f"{summary['balanced_shifted_verdict_agreement']}"
            f"/{summary['instances']} "
            f"bitwise={summary['bitwise_equal_to_recorded']} "
            f"maxdiff={summary['maximum_maximum_velocity_difference']:.3e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
