#!/usr/bin/env python3
"""Execute one digest-bound exact-blind SIAM CHTC action."""
from pathlib import Path
import sys
from pipelines.time_dynamics.siam_chtc_package import run_prepared_siam_action

PACKAGE_DIR = Path('chtc/siam_nb4_avqds_screen_v3_source_39cd411e_20260827')
ACTION_ID = 'siam-nb4-avqds-screen-r00-cut1em03'
ACTION_SHA256 = 'c5c7b9a911474166a6fa216bd74924bacc78f8ad8cd17d6906d61703178ba917'
SOURCE_REVISION = '39cd411efe18a5605675876655c8d7b9fa01b579'

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] != ACTION_ID:
        raise SystemExit(f'usage: run_action.py {ACTION_ID}')
    raise SystemExit(run_prepared_siam_action(
        package_dir=PACKAGE_DIR,
        expected_action_id=ACTION_ID,
        expected_action_sha256=ACTION_SHA256,
        expected_source_revision=SOURCE_REVISION,
    ))
