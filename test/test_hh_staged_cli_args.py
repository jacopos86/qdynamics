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


def test_staged_parser_keeps_phase1_v1_compatibility_default() -> None:
    args = parse_args(["--L", "2", "--skip-pdf"])
    assert str(args.adapt_continuation_mode) == "phase1_v1"
    assert str(args.replay_continuation_mode) == "auto"
    assert str(args.phase3_oracle_gradient_mode) == "off"
    assert str(args.phase3_oracle_execution_surface) == "auto"
    assert str(args.phase3_oracle_raw_transport) == "auto"


def test_staged_parser_accepts_phase3_raw_oracle_args() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--skip-pdf",
            "--adapt-continuation-mode",
            "phase3_v1",
            "--phase3-oracle-gradient-mode",
            "backend_scheduled",
            "--phase3-oracle-use-fake-backend",
            "--phase3-oracle-backend-name",
            "FakeNighthawk",
            "--phase3-oracle-execution-surface",
            "raw_measurement_v1",
            "--phase3-oracle-raw-transport",
            "sampler_v2",
            "--phase3-oracle-raw-store-memory",
        ]
    )
    assert str(args.phase3_oracle_gradient_mode) == "backend_scheduled"
    assert bool(args.phase3_oracle_use_fake_backend) is True
    assert str(args.phase3_oracle_execution_surface) == "raw_measurement_v1"
    assert str(args.phase3_oracle_raw_transport) == "sampler_v2"
    assert bool(args.phase3_oracle_raw_store_memory) is True


def test_checkpoint_controller_mode_accepts_exact_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "exact_v1"])
    assert str(args.checkpoint_controller_mode) == "exact_v1"


def test_checkpoint_controller_mode_accepts_oracle_v1() -> None:
    args = parse_args(["--L", "2", "--skip-pdf", "--checkpoint-controller-mode", "oracle_v1"])
    assert str(args.checkpoint_controller_mode) == "oracle_v1"
