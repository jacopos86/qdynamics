from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_staged_noiseless import parse_args


def test_checkpoint_controller_mode_defaults_off() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.checkpoint_controller_mode) == "off"


def test_checkpoint_controller_mode_accepts_exact_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "exact_v1"])
    assert str(args.checkpoint_controller_mode) == "exact_v1"


def test_checkpoint_controller_mode_accepts_oracle_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "oracle_v1"])
    assert str(args.checkpoint_controller_mode) == "oracle_v1"
