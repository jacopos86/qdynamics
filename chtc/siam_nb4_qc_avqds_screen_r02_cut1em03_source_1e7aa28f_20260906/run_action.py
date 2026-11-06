#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r02_cut1em03_source_1e7aa28f_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r02-cut1em03'
ACTION_SHA256 = 'fa0c34531c003eb0a1a1f2cbb98e226157132bcd7669acc8522a2edc72b4a13c'
SOURCE_REVISION = '1e7aa28f58e23df396a0ac5cf7fa766a001cbd55'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
