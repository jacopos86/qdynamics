#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_avqds_screen_ridge_sigma_v2_rebaseline_v5_source_74da5166_20260828')
ACTION_ID = 'siam-nb4-avqds-screen-r00-cut1em03'
ACTION_SHA256 = 'fa1300f89088fb32708db8f7c25b636e818c781923950cd41ae8fa6556aff379'
SOURCE_REVISION = '74da516646798e2243934bed89d8f65ccd6543d4'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
