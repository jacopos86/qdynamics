#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r00_cut1em02_source_a3d1cebe_20260905')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r00-cut1em02'
ACTION_SHA256 = 'b1e20bcfc770f76f70c60152a5af6e0128da436cc7cdfe0843dd4593a62ad23a'
SOURCE_REVISION = 'a3d1cebe863b92e97d10d2c8bffea2f6bf55d5f8'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
