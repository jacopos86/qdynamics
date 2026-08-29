#!/usr/bin/env python3
"""Run the declared Paper-III base/completion fixed-pool experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.qse_spectra.matched_route_acquisition import (
    ensure_hh_pool_profile_ablation_campaign,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Versioned resumable workspace for both declared pool profiles.",
    )
    args = parser.parse_args(argv)
    payload = ensure_hh_pool_profile_ablation_campaign(args.output_root)
    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "selector_policy_id": payload["selector_policy_id"],
                "pool_policy_ids": payload["pool_policy_ids"],
                "lock_gate": payload["lock_gate"],
                "output_root": str(args.output_root),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
