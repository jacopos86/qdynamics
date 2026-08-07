#!/usr/bin/env python3
"""Validate the Phase-III weak-weak tau=1e-6 confirmation activation."""

from __future__ import annotations

import os
from pathlib import Path
import sys


ACTIVATION_DIR = Path(__file__).resolve().parent
if str(ACTIVATION_DIR) not in sys.path:
    sys.path.insert(0, str(ACTIVATION_DIR))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

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

