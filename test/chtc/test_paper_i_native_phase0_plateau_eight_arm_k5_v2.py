from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_paper_i_native_phase0_plateau_eight_arm_k5_20260816_v2.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "paper_i_native_phase0_eight_arm_v2", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_preserves_the_exact_eight_arm_order_and_isolates_v1_artifacts() -> None:
    runner = _load_runner()

    assert runner.CAMPAIGN_ID.endswith("_v2")
    assert len(runner.CELL_SPECS) == 8
    assert [
        (cell.placement, cell.score, cell.cardinality)
        for cell in runner.CELL_SPECS
    ] == [
        ("generator_first", "gradient", "fixed24"),
        ("position_aware", "gradient", "fixed24"),
        ("generator_first", "gradient", "adaptive"),
        ("position_aware", "gradient", "adaptive"),
        ("generator_first", "proxy", "fixed24"),
        ("position_aware", "proxy", "fixed24"),
        ("generator_first", "proxy", "adaptive"),
        ("position_aware", "proxy", "adaptive"),
    ]
    assert all(cell.regime_id == "strong_weak_u8" for cell in runner.CELL_SPECS)
    assert all(cell.nph == 3 and cell.horizon == 5 for cell in runner.CELL_SPECS)
    assert all(cell.insertion_policy == "plateau_commutation" for cell in runner.CELL_SPECS)
    assert runner.LEGACY_V1_RUNNER_PATH != runner.RUNNER_PATH
