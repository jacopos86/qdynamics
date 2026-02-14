"""Thin shim to canonical ``src.quantum.pauli_letters_module`` with standalone fallback."""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from src.quantum.pauli_letters_module import PauliLetter, symbol_product_map  # noqa: F401, E402
except Exception:
    from ._standalone_pauli_letters_module import PauliLetter, symbol_product_map  # noqa: F401

__all__ = ["PauliLetter", "symbol_product_map"]
