#!/usr/bin/env python3
"""Read-only validation for the Append-ADAPT r70 activation overlay."""

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
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
