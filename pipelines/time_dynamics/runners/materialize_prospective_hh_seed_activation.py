"""CLI for the local-only prospective Paper-II HH seed activation boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipelines.time_dynamics.prospective_hh_seed_activation import (
    materialize_hh_seed_activation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize exactly six canonical HH seed execution requests and "
            "prepared work orders without executing VQE or authorizing submission."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-epoch-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_hh_seed_activation(
        repository_root=Path(args.repo_root),
        source_epoch_receipt=args.source_epoch_receipt,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "PREPARED_NOT_EXECUTED",
                "manifest": result.manifest_path,
                "manifest_file_sha256": result.manifest_file_sha256,
                "manifest_sha256": result.manifest_sha256,
                "source_epoch_identity_sha256": (
                    result.source_epoch_receipt.source_epoch.identity_sha256
                ),
                "action_count": len(result.actions),
                "production_submission_enabled": False,
                "execution_authorized": False,
                "vqe_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
