#!/usr/bin/env python3
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.reporting.hh_driven_dynamics_comparison_report import *  # noqa: F401,F403
from pipelines.reporting.hh_driven_dynamics_comparison_report import main


if __name__ == "__main__":
    raise SystemExit(main())
