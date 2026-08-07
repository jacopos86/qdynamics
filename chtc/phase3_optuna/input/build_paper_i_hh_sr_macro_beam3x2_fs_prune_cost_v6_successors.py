#!/usr/bin/env python3
"""Build immutable v6 successors after the v5 fail-closed review."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_PATH = Path(__file__).with_name(
    "build_paper_i_hh_sr_macro_beam3x2_fs_prune_cost_v5_successors.py"
)
_SPEC = importlib.util.spec_from_file_location("_sr_macro_cost_successor_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("unable to load shared successor builder")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

_BASE.SUCCESSOR_REVISION = "v6"
_BASE.VALIDATOR_REVISION = "v6"
_BASE.CREATED_UTC = "2026-07-20T03:30:00Z"


def main() -> None:
    _BASE.main()


if __name__ == "__main__":
    main()
