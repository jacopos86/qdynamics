"""Thin re-export shim for canonical Hartree-Fock helpers in ``src.quantum``."""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.quantum.hartree_fock_reference_state import *  # noqa: F401,F403,E402
