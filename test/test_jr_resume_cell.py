from __future__ import annotations

import pytest

from chtc.phase3_optuna.run_jr_resume_cell import _replace_flag_value


def test_replace_flag_value_changes_only_requested_controller_cap() -> None:
    argv = [
        "run",
        "--max-depth",
        "50",
        "--phase1-shortlist-size",
        "32",
    ]

    _replace_flag_value(argv, "--max-depth", 5)

    assert argv == [
        "run",
        "--max-depth",
        "5",
        "--phase1-shortlist-size",
        "32",
    ]


def test_replace_flag_value_fails_closed_when_flag_is_missing() -> None:
    with pytest.raises(ValueError, match="missing required flag"):
        _replace_flag_value(["run"], "--max-depth", 5)
