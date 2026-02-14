"""Thin re-export shim to canonical ``src.quantum`` PauliTerm."""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.quantum.qubitization_module import PauliTerm  # noqa: F401, E402

__all__ = ["PauliTerm"]
