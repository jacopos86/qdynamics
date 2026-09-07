#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r08_cut1em06_source_fd505746_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r08-cut1em06'
ACTION_SHA256 = '31822073dfc25baf6257ea6b7e290e67b76b7d3b67875bfb556132ec6a1b51a8'
SOURCE_REVISION = 'fd505746fb1ef29310699bee1df66eb62afb2e8b'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
