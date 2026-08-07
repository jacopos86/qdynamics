#!/usr/bin/env python3
"""Read-only validation for the authorized sealed-v2 execution overlay."""

from __future__ import annotations

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
)


def main() -> int:
    result = validate_activation(repo_root_from_script(__file__))
    manifest = result["manifest"]
    print(
        canonical_json_bytes(
            {
                "status": "passed",
                "activation_id": manifest["activation_id"],
                "activation_manifest_sha256": manifest["sha256"],
                "batch_name": manifest["batch_name"],
                "direct_execution_count": manifest[
                    "direct_execution_count"
                ],
                "execution_authorized": True,
                "submission_authorized": True,
                "submission_state": "authorized_not_submitted",
                "remote_stage": False,
                "condor_submit": False,
                "submitted": False,
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
