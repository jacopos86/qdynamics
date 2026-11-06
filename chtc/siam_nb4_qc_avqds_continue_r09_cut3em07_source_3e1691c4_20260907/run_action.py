#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_continue_r09_cut3em07_source_3e1691c4_20260907')
ACTION_ID = 'siam-nb4-qc-avqds-continue-r09-cut3em07'
ACTION_SHA256 = '917002670a5c7ffcdc0da20476a03a592570b50b638c152ddd1ac51a652e1eb3'
SOURCE_REVISION = '3e1691c4acdf7ad6af05185455626d2166a5550e'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
