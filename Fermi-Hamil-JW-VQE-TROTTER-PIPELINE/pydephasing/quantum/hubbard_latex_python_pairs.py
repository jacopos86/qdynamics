"""Compatibility shim for Hubbard helpers.

Canonical implementation lives in ``src.quantum.hubbard_latex_python_pairs``.
When ``src`` is unavailable (standalone imported repo usage), we fall back to
local implementations.
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from src.quantum.hubbard_latex_python_pairs import *  # noqa: F401,F403,E402
except Exception:
    from ._standalone_hubbard_latex_python_pairs import *  # noqa: F401,F403
