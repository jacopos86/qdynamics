#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_full_apm_screen_r00_cut1em02_source_2163171b_20260905')
ACTION_ID = 'siam-nb4-qc-full_apm-screen-r00-cut1em02'
ACTION_SHA256 = '091a9e44aa10d7b0a866b2879519f29c7afb6fec15e31738a07035f9e75b102c'
SOURCE_REVISION = '2163171bbbf98015fe741b445fc9e1dd3e1794aa'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
