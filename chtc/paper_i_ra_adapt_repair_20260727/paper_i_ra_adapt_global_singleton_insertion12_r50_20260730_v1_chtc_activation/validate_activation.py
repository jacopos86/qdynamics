#!/usr/bin/env python3
"""Read-only validation for the authorized 12-cell execution overlay."""

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
    validate_remote_expanded_dry_run,
    validate_remote_factory_dry_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote-expanded-dry-run",
        "--remote-dry-run",
        dest="remote_expanded_dry_run",
        help=(
            "Validate the 12-ad remote dry-run generated from the "
            "nonfactory submit projection. --remote-dry-run is retained "
            "as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--remote-factory-dry-run",
        help=(
            "Validate the one-cluster-ad remote dry-run generated from "
            "the sealed factory submit description."
        ),
    )
    args = parser.parse_args()
    if bool(args.remote_expanded_dry_run) != bool(
        args.remote_factory_dry_run
    ):
        parser.error(
            "--remote-expanded-dry-run (or --remote-dry-run) and "
            "--remote-factory-dry-run are required together"
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
        "resource_status": manifest["resource_status"],
        "execution_authorized": True,
        "submission_authorized": True,
        "submission_state": "authorized_not_submitted",
        "remote_stage": False,
        "condor_submit": False,
        "submitted": False,
    }
    if args.remote_expanded_dry_run:
        expanded_dry_run = validate_remote_expanded_dry_run(
            repo_root,
            args.remote_expanded_dry_run,
        )
        factory_dry_run = validate_remote_factory_dry_run(
            repo_root,
            args.remote_factory_dry_run,
        )
        output["remote_expanded_dry_run_validation"] = {
            "status": expanded_dry_run["status"],
            "sha256": expanded_dry_run["sha256"],
            "kind": expanded_dry_run["kind"],
            "ad_count": expanded_dry_run["ad_count"],
            "proc_ids": expanded_dry_run["proc_ids"],
            "leave_in_queue": expanded_dry_run["leave_in_queue"],
            "post_submit_factory_expectations": (
                expanded_dry_run["post_submit_factory_expectations"]
            ),
        }
        output["remote_factory_dry_run_validation"] = {
            "status": factory_dry_run["status"],
            "sha256": factory_dry_run["sha256"],
            "kind": factory_dry_run["kind"],
            "cluster_ad_count": factory_dry_run["cluster_ad_count"],
            "cluster_id": factory_dry_run["cluster_id"],
            "batch_name": factory_dry_run["batch_name"],
            "leave_in_queue": factory_dry_run["leave_in_queue"],
            "post_submit_factory_expectations": (
                factory_dry_run["post_submit_factory_expectations"]
            ),
        }
    print(canonical_json_bytes(output).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
