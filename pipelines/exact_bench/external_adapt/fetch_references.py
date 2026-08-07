#!/usr/bin/env python3
"""CLI for listing or fetching cataloged external ADAPT reference repos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.exact_bench.external_adapt.provenance import get_external_reference_spec, reference_catalog
from pipelines.exact_bench.external_adapt.repository_manager import (
    catalog_payload,
    materialize_reference,
    write_reference_lock,
)


def _reference_ids(values: Sequence[str] | None, *, all_references: bool) -> tuple[str, ...]:
    if all_references:
        return tuple(spec.reference_id for spec in reference_catalog() if spec.clone_url is not None)
    return tuple(str(v) for v in (values or ()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List or fetch external ADAPT benchmark references.")
    parser.add_argument("--list", action="store_true", help="Print the catalog and exit.")
    parser.add_argument("--reference-id", action="append", default=None, help="Catalog reference ID to fetch.")
    parser.add_argument("--all", action="store_true", help="Fetch all public-git catalog entries.")
    parser.add_argument("--cache-root", type=Path, default=None, help="External checkout cache root.")
    parser.add_argument("--ref", default=None, help="Override git ref for all selected references.")
    parser.add_argument("--no-fetch", action="store_true", help="Do not git fetch existing checkouts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(json.dumps(catalog_payload(), indent=2, sort_keys=True))
        return 0
    ids = _reference_ids(args.reference_id, all_references=bool(args.all))
    if not ids:
        raise SystemExit("select --reference-id, --all, or --list")
    materialized = []
    for reference_id in ids:
        # Validate early so misspelled IDs fail before any clone starts.
        get_external_reference_spec(reference_id)
    for reference_id in ids:
        materialized.append(
            materialize_reference(
                reference_id,
                cache_root=args.cache_root,
                ref=args.ref,
                fetch=not bool(args.no_fetch),
            )
        )
    lock_path = write_reference_lock(materialized, cache_root=args.cache_root)
    print(json.dumps({"lock_path": str(lock_path), "references": [m.to_dict() for m in materialized]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
