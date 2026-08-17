from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_weak_append_only_first2_k50_20260816.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("paper_i_append2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_two_targets_are_first_append_only_k50():
    module = _module()
    runner = module._load_runner()
    targets = module._target_ids(runner)
    assert targets == (
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_append_only",
        "global_singleton_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__nph3__ra_global_singleton_gradient_phase0_phase123_qiskit_phase23_append_only",
    )
    assert runner.TARGET_HORIZON == 50
    assert targets == tuple(cell.execution_id for cell in runner.TARGET_CELLS[:2])


def test_subset_excludes_other_authorized_cells():
    module = _module()
    runner = module._load_runner()
    targets = set(module._target_ids(runner))
    assert len(targets) == 2
    assert targets.isdisjoint(set(runner.TARGET_EXECUTION_IDS[2:]))
