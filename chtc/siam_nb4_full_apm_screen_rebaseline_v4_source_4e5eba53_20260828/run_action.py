#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_full_apm_screen_rebaseline_v4_source_4e5eba53_20260828')
ACTION_ID = 'siam-nb4-full_apm-screen-r00-cut1em03'
ACTION_SHA256 = 'b8432f2537dbdd369dcf1457def0235bed0098b500fa46754cecb79a13eb8c82'
SOURCE_REVISION = '4e5eba53ed98d13a4213bd4699d7a0922d088b1a'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
