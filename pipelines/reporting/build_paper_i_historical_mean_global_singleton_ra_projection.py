#!/usr/bin/env python3
"""Build one portable page-7 RA projection beside a preserved CHTC archive.

Run this on an execute node (or another host with the complete archive), not on
the CHTC access point.  The output contains the fully validated trajectory and
the two RA prefix-cost observations required by the six-panel page, while the
multi-gigabyte attempt archive remains at its authenticated remote path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (
    add_paper_i_historical_mean_global_singleton_full6_page as page7,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--append-adapter", type=Path, required=True)
    result.add_argument("--regime", choices=tuple(page7.REGIME_ORDER), required=True)
    result.add_argument("--archive", type=Path, required=True)
    result.add_argument("--archive-validation", type=Path, required=True)
    result.add_argument("--remote-archive-path", required=True)
    result.add_argument("--remote-archive-sha256", required=True)
    result.add_argument("--remote-archive-size-bytes", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        projection = page7.build_compact_ra_projection(
            append_adapter_path=args.append_adapter.resolve(),
            regime=args.regime,
            archive_path=args.archive.resolve(),
            archive_validation_path=args.archive_validation.resolve(),
            remote_archive_path=args.remote_archive_path,
            remote_archive_sha256=args.remote_archive_sha256,
            remote_archive_size_bytes=args.remote_archive_size_bytes,
            output=args.output.resolve(),
        )
    except (OSError, RuntimeError, ValueError, page7.Page7InputError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        json.dumps(
            {
                "status": "passed",
                "regime_id": projection["regime_id"],
                "execution_id": projection["execution_id"],
                "projection_sha256": projection["sha256"],
                "source_archive": projection["source_archive"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
