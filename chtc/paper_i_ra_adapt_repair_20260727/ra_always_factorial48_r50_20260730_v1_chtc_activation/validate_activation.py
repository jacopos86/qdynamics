#!/usr/bin/env python3
"""Read-only validation for the authorized 48-cell execution overlay."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ACTIVATION_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))

from activation_contract import (  # noqa: E402
    canonical_json_bytes,
    repo_root_from_script,
    validate_activation,
    validate_remote_factory_dry_run,
    validate_remote_dry_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-expanded-dry-run",
        "--remote-dry-run",
        dest="remote_expanded_dry_run",
        help=(
            "Validate all ProcId ads from the sealed nonfactory expansion "
            "projection."
        ),
    )
    parser.add_argument(
        "--remote-factory-dry-run",
        help="Validate the real factory condor_submit -dry-run cluster ad.",
    )
    args = parser.parse_args()
    if bool(args.remote_expanded_dry_run) != bool(
        args.remote_factory_dry_run
    ):
        parser.error(
            "Remote preflight requires both the expanded projection and "
            "factory dry-run."
        )

    repo_root = repo_root_from_script(__file__)
    result = validate_activation(repo_root)
    manifest = result["manifest"]
    output = {
        "status": "passed",
        "activation_id": manifest["activation_id"],
        "activation_manifest_sha256": manifest["sha256"],
        "batch_name": manifest["batch_name"],
        "direct_execution_count": manifest["direct_execution_count"],
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if args.remote_expanded_dry_run:
        expanded = validate_remote_dry_run(
            repo_root, args.remote_expanded_dry_run
        )
        factory = validate_remote_factory_dry_run(
            repo_root, args.remote_factory_dry_run
        )
        output["remote_dry_run_validation"] = {
            "status": "passed",
            "expanded_nonfactory_projection": {
                "status": expanded["status"],
                "sha256": expanded["sha256"],
                "ad_count": expanded["ad_count"],
                "proc_ids": expanded["proc_ids"],
                "observed_leave_in_queue": expanded[
                    "observed_leave_in_queue"
                ],
            },
            "factory_cluster_ad": {
                "status": factory["status"],
                "sha256": factory["sha256"],
                "cluster_id": factory["cluster_id"],
                "observed_leave_in_queue": factory[
                    "observed_leave_in_queue"
                ],
            },
            "live_factory_query_required": True,
            "live_factory_expected_attributes": factory[
                "live_factory_expected_attributes"
            ],
        }
    print(canonical_json_bytes(output).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
