#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r01_cut3em03_source_2c25b0bf_20260905')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r01-cut3em03'
ACTION_SHA256 = 'e8c25fc14fe777c6a84a64efc8c300642916297447bf657fa90b3a2e5bc89af3'
SOURCE_REVISION = '2c25b0bf7cb13d0fd0d0d034d6d4ea80ffae8507'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
