#!/usr/bin/env python3
"""Read-only validation for the inert fresh round-100 Append package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))
sys.dont_write_bytecode = True

from package_contract import (  # noqa: E402
    PackageContractError,
    canonical_json_bytes,
    validate_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-archive-scan", action="store_true")
    parser.add_argument("--full-anchor-scan", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = validate_package(
            full_archive_scan=args.full_archive_scan,
            full_anchor_scan=args.full_anchor_scan,
        )
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
