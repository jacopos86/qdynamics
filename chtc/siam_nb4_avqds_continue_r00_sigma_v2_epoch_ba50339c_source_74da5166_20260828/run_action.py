#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_avqds_continue_r00_sigma_v2_epoch_ba50339c_source_74da5166_20260828')
ACTION_ID = 'siam-nb4-avqds-continue-r00-cut1em03'
ACTION_SHA256 = '014d41e96ba3d3db79eb38ea241c7b8219123a7e695d6113b72930ee82bc7f7f'
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
