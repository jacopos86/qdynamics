#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r04_cut1em04_source_4c7405bf_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r04-cut1em04'
ACTION_SHA256 = '822b4b35375b80ba4fb07f1d4ca2a5833c79fa2c679e8ef2aaf1c35a320da2ef'
SOURCE_REVISION = '4c7405bfc8d4f8568329668e4abea15f0ba0dae4'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
