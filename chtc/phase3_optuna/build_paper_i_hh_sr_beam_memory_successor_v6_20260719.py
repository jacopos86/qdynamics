#!/usr/bin/env python3
"""Build the immutable six-row memory successor for historical beam3x2.

This wrapper reuses the established operational-only builder and changes only
the successor identity plus Condor memory requests.  Scientific manifests are
still compared against the byte-identical v4 parent after operational fields
are removed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name(
    "build_paper_i_hh_sr_beam_memory_successor_20260719.py"
)
SPEC = importlib.util.spec_from_file_location("beam_memory_successor_base", SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(BASE)

BASE.SUCCESSOR_ID = (
    "paper_i_hh_sr_snake_appendix_historical_beam3x2_full_response_"
    "symmetric_cost_noprune_no_ordinary_novelty_all_six_memory_successor_"
    "r50_20260719_v6_chtc"
)
BASE.SUCCESSOR_BATCH = (
    "paper-i-hh-sr-appendix-historical-beam3x2-memory-repair-six-"
    "r50-20260719-v6"
)
BASE.BUILDER_SCRIPT_NAME = Path(__file__).name
BASE.RETRY_SCOPE = "all_six_memory_held_rows_after_v4_v5_failures"
BASE.UNAFFECTED_PARENT_PROCS = []

# old_memory_mb is the exact v4 parent-manifest value used by the scientific
# equality check.  failure_* fields identify the live row that established the
# latest observed memory tier.  New requests follow the repository's fixed
# ceil(1.10 * MemoryUsage / 4096) * 4096 policy.
BASE.ROWS = (
    {"proc": 0, "slug": "weak_weak", "old_memory_mb": 32768,
     "observed_memory_usage_mb": 41504, "resident_set_size_raw": 41567296,
     "image_size": 42500000, "new_memory_mb": 49152, "disk_mb": 61440,
     "failure_cluster": 8887761, "failure_proc": 0,
     "failed_request_memory_mb": 40960},
    {"proc": 1, "slug": "intermediate_weak", "old_memory_mb": 32768,
     "observed_memory_usage_mb": 41504, "resident_set_size_raw": 41552224,
     "image_size": 42500000, "new_memory_mb": 49152, "disk_mb": 61440,
     "failure_cluster": 8887761, "failure_proc": 1,
     "failed_request_memory_mb": 40960},
    {"proc": 2, "slug": "strong_weak_u8", "old_memory_mb": 40960,
     "observed_memory_usage_mb": 48829, "resident_set_size_raw": 49925828,
     "image_size": 50000000, "new_memory_mb": 57344, "disk_mb": 61440,
     "failure_cluster": 8887761, "failure_proc": 2,
     "failed_request_memory_mb": 49152},
    {"proc": 3, "slug": "weak_strong", "old_memory_mb": 49152,
     "observed_memory_usage_mb": 48829, "resident_set_size_raw": 48768096,
     "image_size": 50000000, "new_memory_mb": 57344, "disk_mb": 81920,
     "failure_cluster": 8887576, "failure_proc": 3,
     "failed_request_memory_mb": 49152},
    {"proc": 4, "slug": "intermediate_strong", "old_memory_mb": 49152,
     "observed_memory_usage_mb": 48829, "resident_set_size_raw": 49921404,
     "image_size": 50000000, "new_memory_mb": 57344, "disk_mb": 81920,
     "failure_cluster": 8887576, "failure_proc": 4,
     "failed_request_memory_mb": 49152},
    {"proc": 5, "slug": "strong_strong_u8", "old_memory_mb": 49152,
     "observed_memory_usage_mb": 73243, "resident_set_size_raw": 58341520,
     "image_size": 75000000, "new_memory_mb": 81920, "disk_mb": 81920,
     "failure_cluster": 8887761, "failure_proc": 3,
     "failed_request_memory_mb": 57344},
)


def build() -> Path:
    return BASE.build()


def verify(successor: Path | None = None) -> None:
    BASE.verify(successor)


if __name__ == "__main__":
    print(build().relative_to(BASE.REPO))
