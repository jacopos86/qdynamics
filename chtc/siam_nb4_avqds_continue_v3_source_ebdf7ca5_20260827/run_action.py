#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_avqds_continue_v3_source_ebdf7ca5_20260827')
ACTION_ID = 'siam-nb4-avqds-continue-r00-cut1em03'
ACTION_SHA256 = '0cd9a0551d074679d2a664226ac005c5f24268dd965c47a90627452146d0e556'
SOURCE_REVISION = 'ebdf7ca593f0ea56e06300341b5d408f7961da00'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
