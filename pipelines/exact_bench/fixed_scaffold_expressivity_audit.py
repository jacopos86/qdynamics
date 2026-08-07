#!/usr/bin/env python3
"""Diagnostic fixed-scaffold expressivity audit for static ADAPT/Route-C work.

This CLI is intentionally diagnostic.  It does not run an adaptive selector and
it does not update paper-facing tables.  It builds a fixed full_meta scaffold,
optionally seeded from an existing ADAPT/Route-C JSON, and refits that identical
scaffold with one or more stronger optimizers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pipelines.exact_bench.generic_static_adapt_variants import (
    run_fixed_scaffold_expressivity_audit_single,
)


def _parse_optimizer_kinds(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, help="Benchmark family, e.g. hh")
    parser.add_argument("--case-id", required=True, help="Canonical case id")
    parser.add_argument(
        "--table-i-suite-profile",
        default=None,
        help="Optional Table-I suite profile for resolving diagnostic cases, e.g. hh_symmetric",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for result.json")
    parser.add_argument(
        "--source-json",
        type=Path,
        default=None,
        help="Optional ADAPT/Route-C JSON whose adapt_vqe scaffold seeds the fixed ansatz",
    )
    parser.add_argument(
        "--warm-start-json",
        type=Path,
        default=None,
        help=(
            "Optional prior fixed-scaffold audit result. Its best theta is copied into the target "
            "scaffold prefix after strict operator-label validation; new coordinates stay zero."
        ),
    )
    parser.add_argument(
        "--theta-coordinate-mode",
        choices=("auto", "logical_shared", "per_pauli_term"),
        default="auto",
        help=(
            "Fixed-scaffold parameterization mode. auto preserves source mode; per_pauli_term gives "
            "pure full-meta scaffolds one runtime parameter per non-identity Pauli term."
        ),
    )
    parser.add_argument(
        "--max-scaffold-terms",
        type=int,
        default=64,
        help="Final fixed-scaffold size after full_meta padding. Source scaffold is never truncated.",
    )
    parser.add_argument(
        "--pool-term-cap",
        type=int,
        default=512,
        help="Maximum full_meta pool terms to build before padding the scaffold.",
    )
    parser.add_argument(
        "--pool-indices",
        default=None,
        help=(
            "Optional comma/range list of full_meta pool indices to force into the fixed scaffold, "
            "for example '0-57,78-102'. Explicit indices are added before prefix padding."
        ),
    )
    parser.add_argument(
        "--optimizer-kinds",
        default="powell,qnspsa",
        help="Comma-separated refit optimizers: powell,qnspsa,geo_qngd,spsa,bfgs",
    )
    parser.add_argument("--optimizer-maxiter", type=int, default=5000)
    parser.add_argument("--metric-floor", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-source-policy",
        choices=("fail", "skip", "ignore"),
        default="fail",
        help="What to do if source scaffold labels are absent from this full_meta pool.",
    )
    parser.add_argument("--same-cutoff-exact-gs-energy", type=float, default=None)
    parser.add_argument("--exact-reference-energy", type=float, default=None)
    parser.add_argument("--exact-reference-n-ph-max", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_fixed_scaffold_expressivity_audit_single(
        family=args.family,
        case_id=args.case_id,
        output_dir=args.output_dir,
        table_i_suite_profile=args.table_i_suite_profile,
        source_json=args.source_json,
        warm_start_json=args.warm_start_json,
        theta_coordinate_mode=args.theta_coordinate_mode,
        pool_indices=args.pool_indices,
        max_scaffold_terms=args.max_scaffold_terms,
        optimizer_kinds=_parse_optimizer_kinds(args.optimizer_kinds),
        optimizer_maxiter=args.optimizer_maxiter,
        metric_floor=args.metric_floor,
        seed=args.seed,
        pool_term_cap=args.pool_term_cap,
        missing_source_policy=args.missing_source_policy,
        same_cutoff_exact_gs_energy=args.same_cutoff_exact_gs_energy,
        exact_reference_energy=args.exact_reference_energy,
        exact_reference_n_ph_max=args.exact_reference_n_ph_max,
    )
    best = payload.get("best_result") or {}
    print(f"wrote {args.output_dir / 'result.json'}")
    if best:
        print(
            "best "
            f"optimizer={best.get('optimizer_kind')} "
            f"energy={float(best.get('energy')):.12e} "
            f"same_cutoff_abs_delta_e={best.get('same_cutoff_abs_delta_e')}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
