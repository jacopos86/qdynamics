from __future__ import annotations

import shlex
import subprocess

from pipelines.time_dynamics.hubbard_apm_operational_launch_recovery import (
    repaired_apptainer_line,
    source_receipt_validation_command,
)


def test_hubbard_source_receipt_command_survives_nested_bash_parsing() -> None:
    command = source_receipt_validation_command()
    probe = command.replace("cd /work && PYTHONPATH=. python3 -c ", "printf '%s' ", 1)
    completed = subprocess.run(
        ["bash", "-lc", probe], check=True, capture_output=True, text=True
    )
    assert "SourceEpochReceipt.load(" in completed.stdout
    assert "repository_root='.'" in completed.stdout
    assert "paper_ii_hubbard_l4_scale_v1" in completed.stdout
    assert "source_epoch_receipt.json" in completed.stdout


def test_hubbard_apptainer_line_has_one_literal_bash_lc_argument() -> None:
    line = repaired_apptainer_line()
    suffix = line.split(" bash -lc ", 1)[1]
    words = shlex.split(f"bash -lc {suffix}")
    assert words[:2] == ["bash", "-lc"]
    assert len(words) == 3
    assert words[2] == source_receipt_validation_command()
