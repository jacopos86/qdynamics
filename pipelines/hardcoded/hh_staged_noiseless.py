#!/usr/bin/env python3
"""Thin CLI wrapper for the staged HH noiseless workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_cli_args import build_staged_hh_parser
from pipelines.hardcoded.hh_staged_workflow import (
    resolve_staged_hh_config,
    run_staged_hh_noiseless,
)


def build_parser() -> argparse.ArgumentParser:
    return build_staged_hh_parser(
        description=(
            "Historical/compatibility staged HH noiseless workflow: HF -> hh_hva_ptw warm-start -> staged ADAPT -> "
            "matched-family conventional replay -> Suzuki/CFQM vs exact. "
            "Kept for VQE->ADAPT->VQE reproduction; new HH default work should use pipelines/hardcoded/adapt_pipeline.py with direct phase3_v1 continuation."
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = resolve_staged_hh_config(args)
    payload = run_staged_hh_noiseless(cfg)
    print(f"workflow_json={payload['artifacts']['workflow']['output_json']}")
    if not bool(cfg.artifacts.skip_pdf):
        print(f"workflow_pdf={payload['artifacts']['workflow']['output_pdf']}")
    print(f"adapt_handoff_json={payload['artifacts']['intermediate']['adapt_handoff_json']}")
    print(f"replay_json={payload['artifacts']['intermediate']['replay_output_json']}")
    if "pareto" in payload.get("artifacts", {}):
        print(f"pareto_rows_json={payload['artifacts']['pareto']['run_rows_json']}")
        print(f"pareto_frontier_json={payload['artifacts']['pareto']['run_frontier_json']}")
        print(f"pareto_rolling_frontier_json={payload['artifacts']['pareto']['rolling_frontier_json']}")


if __name__ == "__main__":
    main()
