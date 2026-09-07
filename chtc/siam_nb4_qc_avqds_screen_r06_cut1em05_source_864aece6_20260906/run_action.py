#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r06_cut1em05_source_864aece6_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r06-cut1em05'
ACTION_SHA256 = '11a4071b7f44e4dea09b76bfffa01e85d14008543082de651b6dcf2c8c9878fd'
SOURCE_REVISION = '864aece6ae5b5c3bac910d2a99a98430482c8bb7'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
