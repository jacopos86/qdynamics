#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_full_apm_screen_v3_source_ebdf7ca5_20260827')
ACTION_ID = 'siam-nb4-full_apm-screen-r00-cut1em03'
ACTION_SHA256 = 'ee8016adece7c66974c5ccd5fe1e169f6f89df6d2a2febfa7fbe210216b74af2'
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
