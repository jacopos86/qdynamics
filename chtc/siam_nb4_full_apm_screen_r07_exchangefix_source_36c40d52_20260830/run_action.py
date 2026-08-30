#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_full_apm_screen_r07_exchangefix_source_36c40d52_20260830')
ACTION_ID = 'siam-nb4-full_apm-screen-r07-cut3em06'
ACTION_SHA256 = '5ac1c473f470cd88c70c3fcd855256ffe033148a78bdbf9e67b6c06f16db703b'
SOURCE_REVISION = '36c40d522a4c5fb4daa57234fbc85e82b692e8a3'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
