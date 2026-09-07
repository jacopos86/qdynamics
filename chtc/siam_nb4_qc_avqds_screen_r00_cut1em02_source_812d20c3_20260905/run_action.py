#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_812d20c3_20260905')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r00-cut1em02'
ACTION_SHA256 = '2408fb44d7750d87f154b8c79933b47f8fd118a7229a5e4615a717f2c12f6990'
SOURCE_REVISION = '812d20c358959b14412cf58bfa1dde8facb679a5'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
