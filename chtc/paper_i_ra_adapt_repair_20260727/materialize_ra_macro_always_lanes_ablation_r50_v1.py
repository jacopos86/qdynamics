#!/usr/bin/env python3
"""Seal the inert twelve-cell macro always-insertion lane ablation for CHTC.

Six regimes x {lanes on, lanes off}, horizon 50. The only ablated axis is the
Phase-I shortlist population: nine physical operator lanes versus one global
ranking. Source-locked to the stationary-core v13 macro always-insertion cells.
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
    LANES_ABLATION_BUNDLE_ID,
    build_lanes_ablation_cell_specs,
    materialize_lanes_ablation_bundle,
)


support = core13.support
MATERIALIZATIONS_ROOT = (
    REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations"
)
MATERIALIZATION_ID = "ra_adapt_macro_always_lanes_ablation_r50_v1"
SOURCE_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_stationary_late_core_v13"
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
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    for required in (SOURCE_LOCKS_INPUT, SOURCE_PROBLEM_BASELINES):
        if not required.is_file():
            raise FileNotFoundError(f"Missing source input: {required}")

    receipt = materialize_lanes_ablation_bundle(
        destination,
        problem_resolver=support._problem_resolver_from(
            _load(SOURCE_PROBLEM_BASELINES)
        ),
        source_locks=_load(SOURCE_LOCKS_INPUT),
        repository_state=support._repository_state(),
        repo_root=REPO_ROOT,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=support._utc_now(),
        verify_source_files=True,
    )
    print(f"bundle_id  = {receipt.bundle_id}")
    print(f"cell_count = {receipt.cell_count}")
    print(f"status     = {receipt.materialization_status}")
    print(f"path       = {receipt.bundle_path}")
    for cell in build_lanes_ablation_cell_specs():
        arm = "lanes_off" if "no_lanes" in cell.algorithm_id else "lanes_on"
        print(f"  {arm:9} {cell.regime_id:20} nph{cell.nph}  k={cell.horizon}")
    if receipt.bundle_id != LANES_ABLATION_BUNDLE_ID:
        raise RuntimeError("Materialized bundle identity drifted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
