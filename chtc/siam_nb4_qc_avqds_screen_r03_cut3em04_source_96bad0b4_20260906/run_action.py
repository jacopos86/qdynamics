#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r03_cut3em04_source_96bad0b4_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r03-cut3em04'
ACTION_SHA256 = '72b8f32be7e541e9c5bbdff79bc4827c7288693d0ae459c22c7057d252492967'
SOURCE_REVISION = '96bad0b4efa4eda3a0ba010e2d2954c529e5c371'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
