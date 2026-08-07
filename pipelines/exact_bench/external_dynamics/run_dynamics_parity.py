#!/usr/bin/env python3
"""CLI for diagnostic-only external dynamics parity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.exact_bench.external_dynamics.adapter import (
    DynamicsParityTolerances,
    run_dynamics_parity_checks,
)


def _parse_dt_scales(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", default="all", help="Comma-separated check IDs or all")
    parser.add_argument("--external-manifest", default=None)
    parser.add_argument("--reference-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-unpinned-local-reference", action="store_true")
    parser.add_argument("--require-diagnostic-pass", action="store_true")
    parser.add_argument("--rhs-limit-dt-scales", default="0.2,0.1,0.05")
    parser.add_argument("--tangent-l2-tol", type=float, default=5.0e-3)
    parser.add_argument("--theta-dot-abs-tol", type=float, default=5.0e-3)
    parser.add_argument("--residual-ratio-abs-tol", type=float, default=5.0e-3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_dynamics_parity_checks(
        checks=args.check,
        output_dir=Path(args.output_dir),
        manifest_path=args.external_manifest,
        reference_root=args.reference_root,
        allow_unpinned_local_reference=bool(args.allow_unpinned_local_reference),
        dt_scales=_parse_dt_scales(args.rhs_limit_dt_scales),
        tolerances=DynamicsParityTolerances(
            tangent_l2_tol=float(args.tangent_l2_tol),
            theta_dot_abs_tol=float(args.theta_dot_abs_tol),
            residual_ratio_abs_tol=float(args.residual_ratio_abs_tol),
        ),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if args.require_diagnostic_pass and payload["summary"].get("passed") is not True:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
