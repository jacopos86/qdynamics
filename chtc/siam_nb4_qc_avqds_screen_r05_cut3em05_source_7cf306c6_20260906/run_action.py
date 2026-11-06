#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r05_cut3em05_source_7cf306c6_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r05-cut3em05'
ACTION_SHA256 = 'd128b51f2d7024d06f53996e0c7586e417d262fc3643fd5d8aed78baa800ccaf'
SOURCE_REVISION = '7cf306c623378623aa3ed8b79e85aaae5393e5e8'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
