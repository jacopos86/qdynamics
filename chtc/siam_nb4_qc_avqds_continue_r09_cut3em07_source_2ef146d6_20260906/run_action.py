#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_qc_avqds_continue_r09_cut3em07_source_2ef146d6_20260906')
ACTION_ID = 'siam-nb4-qc-avqds-continue-r09-cut3em07'
ACTION_SHA256 = '0e39decc6ff2d3bc42b9edd3d011a2326fff66f76184f726789931aeb4355e37'
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
