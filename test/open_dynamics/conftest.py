from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER5_SRC = REPO_ROOT / "paper_5" / "src"
for path in (PAPER5_SRC, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def dimer_protocol_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "riva_2026_dimer_protocol_v1.json"
    )


@pytest.fixture(scope="session")
def paper5_vertical_slice(dimer_protocol_path: Path):
    from pipelines.open_dynamics import run_paper5_vertical_slice

    return run_paper5_vertical_slice(
        dimer_protocol_path,
        code_revision="test-fixture",
    )
