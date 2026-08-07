#!/usr/bin/env python3
"""Thin CLI wrapper for HH backend-aware Phase 3 ADAPT on one backend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded import adapt_pipeline as adapt_mod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the heavy HH Phase 3 ADAPT search with a single backend-conditioned "
            "transpilation oracle. Remaining arguments are forwarded to adapt_pipeline.py."
        )
    )
    p.add_argument("--backend-name", required=True, help="Target backend name, e.g. ibm_boston or ibm_miami.")
    p.add_argument("--backend-transpile-seed", type=int, default=7)
    p.add_argument("--backend-optimization-level", type=int, default=1)
    return p


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, forwarded = parse_args(argv)
    forwarded_args = [str(x) for x in forwarded]
    forwarded_args.extend(
        [
            "--problem",
            "hh",
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-backend-cost-mode",
            "transpile_single_v1",
            "--phase3-backend-name",
            str(args.backend_name),
            "--phase3-backend-transpile-seed",
            str(int(args.backend_transpile_seed)),
            "--phase3-backend-optimization-level",
            str(int(args.backend_optimization_level)),
        ]
    )
    adapt_mod.main(forwarded_args)


if __name__ == "__main__":
    main()
