#!/usr/bin/env python3
"""Acquire both locked Paper-III routes for one declared HH case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.qse_spectra.matched_route_acquisition import (
    acquire_selected_vs_direct_hh_case,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = acquire_selected_vs_direct_hh_case(args.case_key, args.output_json)
    print(
        json.dumps(
            {
                "case_key": bundle["case"]["case_key"],
                "methods": sorted(bundle["routes"]),
                "output_json": str(args.output_json),
                "profile_digest": bundle["profile_digest"],
                "source_lock_digest": bundle["source_lock_digest"],
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
