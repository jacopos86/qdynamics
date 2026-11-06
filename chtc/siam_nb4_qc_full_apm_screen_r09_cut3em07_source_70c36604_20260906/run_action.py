#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_full_apm_screen_r09_cut3em07_source_70c36604_20260906')
ACTION_ID = 'siam-nb4-qc-full_apm-screen-r09-cut3em07'
ACTION_SHA256 = '82041ed8128c45492787e2b39db31461b9784cfb28ca48490ae0d3afa2e3c07f'
SOURCE_REVISION = '70c36604a0621e317c4440d76edbbf101f3f2d0d'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
