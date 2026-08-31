#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_avqds_screen_r00_exchangefix_source_36c40d52_20260828')
ACTION_ID = 'siam-nb4-avqds-screen-r00-cut1em02'
ACTION_SHA256 = '980807502edd2a18b3e7b3e4482de933bbd0ad0421eee1f8963a5c2ea9a8bbcd'
SOURCE_REVISION = '36c40d522a4c5fb4daa57234fbc85e82b692e8a3'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
