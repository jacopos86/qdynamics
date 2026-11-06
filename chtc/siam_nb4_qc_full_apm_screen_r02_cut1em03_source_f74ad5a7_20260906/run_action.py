#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_full_apm_screen_r02_cut1em03_source_f74ad5a7_20260906')
ACTION_ID = 'siam-nb4-qc-full_apm-screen-r02-cut1em03'
ACTION_SHA256 = '61c870ca441e3c79df4fff2efabd50e9fa606194d112b23f1ccb3db434d5df80'
SOURCE_REVISION = 'f74ad5a720d19d57bcdd9d502f64108467f2bffc'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
