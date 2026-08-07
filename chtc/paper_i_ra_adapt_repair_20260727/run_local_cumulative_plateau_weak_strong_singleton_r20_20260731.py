#!/usr/bin/env python3
"""Run the weak--strong singleton cumulative-relative plateau diagnostic."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    run_local_cumulative_plateau_pair_20260731 as base,
)


CELL_ID = "core__weak_strong__nph7__ra_singleton_plateau"
OUTPUT_ROOT = (
    REPO_ROOT
    / "output/local_runs"
    / "paper_i_ra_adapt_cumulative_plateau_weak_strong_singleton_r20_local_20260731_v1"
)
EXACT_ENERGY = -1.138720638075003


def _configure() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.MATERIALIZATION_ROOT = OUTPUT_ROOT / "materialization"
    base.RUNS_ROOT = OUTPUT_ROOT / "runs"
    base.DIAGNOSTIC_BUNDLE_ID = (
        "paper_i_ra_adapt_cumulative_relative_plateau_"
        "weak_strong_singleton_r20_local_v1"
    )
    base.MAXIMUM_CONTROLLER_ROUNDS = 20
    base.EXACT_ENERGIES = {CELL_ID: EXACT_ENERGY}
    base.CELLS = (CELL_ID,)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--materialize", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--execution-authorized", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    _configure()
    if args.materialize:
        if args.execution_authorized:
            raise base.DiagnosticContractError(
                "Materialization cannot carry execution authorization."
            )
        print(base._canonical_bytes(base.materialize()).decode("utf-8"))
        return 0
    if not args.execution_authorized:
        raise base.DiagnosticContractError(
            "Execution requires --execution-authorized."
        )
    print(base._canonical_bytes(base.run_cell(CELL_ID)).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
