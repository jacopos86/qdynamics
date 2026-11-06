#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_full_apm_screen_r01_cut3em03_source_2ef146d6_20260906')
ACTION_ID = 'siam-nb4-qc-full_apm-screen-r01-cut3em03'
ACTION_SHA256 = '6f9d814256b492121429f1d0febd56d89e2e140a027ead93af21e0664fad53fd'
SOURCE_REVISION = '2ef146d65907f0b7dbd4dfe150cac8992c0dcffc'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
