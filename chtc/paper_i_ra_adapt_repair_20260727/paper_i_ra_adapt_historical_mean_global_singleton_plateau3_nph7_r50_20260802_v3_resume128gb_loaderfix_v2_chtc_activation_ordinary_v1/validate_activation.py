#!/usr/bin/env python3
"""Read-only validation for the three authorized accepted-state resumes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ACTIVATION_DIR = Path(__file__).resolve().parent
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from activation_contract import (  # noqa: E402
    ActivationContractError,
    canonical_json_bytes,
    repo_root_from_script,
    validate_activation,
)


def main() -> int:
    try:
        result = validate_activation(repo_root_from_script(__file__))
    except (OSError, ActivationContractError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
