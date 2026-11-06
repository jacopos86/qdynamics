#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_screen_r07_cut3em06_source_34f89a50_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-screen-r07-cut3em06'
ACTION_SHA256 = '94d46069a5528a254abdba5f49b865f484c8043d4a75f75ee041e59a94542bec'
SOURCE_REVISION = '34f89a5028cab76f25f03f84cf94a81765d600a4'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
