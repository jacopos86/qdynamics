#!/usr/bin/env python3
"""Seal revision 2 of the source-locked Qiskit-cost always13 diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (
    materialize_ra_qiskit_cost_macro_always13_local_v1 as implementation,
)


MATERIALIZATION_ID = "ra_adapt_qiskit_cost_macro_always13_local_v2"


if __name__ == "__main__":
    raise SystemExit(
        implementation.main(materialization_id=MATERIALIZATION_ID)
    )
