"""Compatibility shim for Hartree-Fock reference helpers.

Canonical implementation lives in ``src.quantum.hartree_fock_reference_state``.
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
    from src.quantum.hartree_fock_reference_state import *  # noqa: F401,F403,E402
except Exception:
    from ._standalone_hartree_fock_reference_state import *  # noqa: F401,F403
