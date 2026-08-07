#!/usr/bin/env python3
"""Seal the inert six-regime macro always-insertion Qiskit-cost diagnostic.

Source-locked to the stationary-core v13 macro always-insertion cells. The
only declared deltas are the compiled selector-cost denominator
(``transpile_single_v1`` instead of the graph-span model), all-phase resource
weighting, and the per-regime controller horizons 20/20/15/20/20/15.

Materialization is inert: ``execution_authorized`` stays false.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    materialize_stationary_core_v13 as core13,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    QISKIT_COST_ALWAYS6_BUNDLE_ID,
    QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME,
    build_qiskit_cost_always6_cell_specs,
    materialize_qiskit_cost_always6_bundle,
)


support = core13.support
MATERIALIZATIONS_ROOT = (
    REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations"
)
MATERIALIZATION_ID = "ra_adapt_qiskit_cost_macro_always6_local_v1"
SOURCE_MATERIALIZATION_ID = "ra_adapt_stationary_late_core_v13"
SOURCE_ROOT = MATERIALIZATIONS_ROOT / SOURCE_MATERIALIZATION_ID
SOURCE_LOCKS_INPUT = SOURCE_ROOT / "source_materialization/source_locks_input.json"
SOURCE_PROBLEM_BASELINES = (
    SOURCE_ROOT / "source_materialization/problem_baselines.json"
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(*, materialization_id: str = MATERIALIZATION_ID) -> int:
    destination = MATERIALIZATIONS_ROOT / materialization_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite always6 materialization: {destination}"
        )
    for required in (SOURCE_LOCKS_INPUT, SOURCE_PROBLEM_BASELINES):
        if not required.is_file():
            raise FileNotFoundError(f"Missing source input: {required}")

    # The v13 lock input is already always-route repaired; re-running the
    # repair would report zero scalar changes and fail its own audit.
    repaired_locks = _load(SOURCE_LOCKS_INPUT)
    baselines = _load(SOURCE_PROBLEM_BASELINES)

    receipt = materialize_qiskit_cost_always6_bundle(
        destination,
        problem_resolver=support._problem_resolver_from(baselines),
        source_locks=repaired_locks,
        repository_state=support._repository_state(),
        repo_root=REPO_ROOT,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=support._utc_now(),
        verify_source_files=True,
    )
    cells = build_qiskit_cost_always6_cell_specs()
    print(f"bundle_id           = {receipt.bundle_id}")
    print(f"cell_count          = {receipt.cell_count}")
    print(f"status              = {receipt.materialization_status}")
    print(f"bundle_path         = {receipt.bundle_path}")
    for cell in cells:
        print(
            f"  {cell.regime_id:20} nph{cell.nph}  "
            f"horizon={QISKIT_COST_ALWAYS6_HORIZON_BY_REGIME[cell.regime_id]}"
        )
    if receipt.bundle_id != QISKIT_COST_ALWAYS6_BUNDLE_ID:
        raise RuntimeError("Materialized bundle identity drifted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
