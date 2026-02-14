"""Thin shim to canonical ``src.quantum.pauli_polynomial_class`` with standalone fallback."""
from __future__ import annotations

import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from src.quantum.pauli_polynomial_class import (  # noqa: F401, E402
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )
except Exception:
    from ._standalone_pauli_polynomial_class import (  # noqa: F401
        PauliPolynomial,
        fermion_minus_operator,
        fermion_plus_operator,
    )

__all__ = ["PauliPolynomial", "fermion_plus_operator", "fermion_minus_operator"]
