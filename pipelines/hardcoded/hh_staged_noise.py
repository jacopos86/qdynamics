#!/usr/bin/env python3
"""Thin CLI wrapper for the staged HH noise-capable workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_cli_args import build_staged_hh_parser
from pipelines.hardcoded.hh_staged_noise_workflow import (
    resolve_staged_hh_noise_config,
    run_staged_hh_noise,
)


def build_parser() -> argparse.ArgumentParser:
    return build_staged_hh_parser(
        description=(
            "Staged HH noise workflow: HF -> hh_hva_ptw warm-start -> staged ADAPT -> "
            "matched-family conventional replay -> final-only noisy/noiseless dynamics."
        ),
        include_noise=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = resolve_staged_hh_noise_config(args)
    payload = run_staged_hh_noise(cfg)
    print(f"workflow_json={payload['artifacts']['workflow']['output_json']}")
    if not bool(cfg.staged.artifacts.skip_pdf):
        print(f"workflow_pdf={payload['artifacts']['workflow']['output_pdf']}")
    print(f"adapt_handoff_json={payload['artifacts']['intermediate']['adapt_handoff_json']}")
    print(f"replay_json={payload['artifacts']['intermediate']['replay_output_json']}")


if __name__ == "__main__":
    main()
