#!/usr/bin/env python3
"""Collect the complete distributed Paper-III selected/direct HH campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.qse_spectra.matched_route_acquisition import (
    collect_selected_vs_direct_hh_cases,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-results-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    case_paths = sorted(args.case_results_dir.glob("*.json"))
    snapshot = collect_selected_vs_direct_hh_cases(case_paths, args.output_root)
    print(json.dumps(snapshot.payload(), indent=1, sort_keys=True))
    return 0 if snapshot.execution_status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
