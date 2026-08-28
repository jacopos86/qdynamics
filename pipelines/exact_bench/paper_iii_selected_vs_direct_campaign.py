#!/usr/bin/env python3
"""Ensure the locked Paper-III selected-QSE/direct-Davidson campaign exists.

The command is idempotent: invoking it again against the same output root
validates and reuses every completed route artifact before executing missing
work.  Its scientific declaration is fixed in code; route selection, hybrid
execution, fallback behavior, and selector tuning are intentionally absent
from the command interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.qse_spectra.matched_route_acquisition import (
    ensure_selected_vs_direct_hh_campaign,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Versioned workspace for checkpoints, evidence, status, and decision artifacts.",
    )
    args = parser.parse_args(argv)
    snapshot = ensure_selected_vs_direct_hh_campaign(args.output_root)
    print(json.dumps(snapshot.payload(), indent=1, sort_keys=True))
    return 0 if snapshot.execution_status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
