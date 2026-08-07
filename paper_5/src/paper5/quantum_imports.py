from __future__ import annotations

from .repo import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.quantum.pauli_polynomial_class import (  # noqa: E402
    PauliPolynomial,
    fermion_minus_operator,
    fermion_plus_operator,
)
from src.quantum.qubitization_module import PauliTerm  # noqa: E402


def jw_creation(n_qubits: int, mode: int) -> PauliPolynomial:
    """Return the repo-standard Jordan-Wigner creation operator."""

    return fermion_plus_operator(repr_mode="JW", nq=int(n_qubits), j=int(mode))


def jw_annihilation(n_qubits: int, mode: int) -> PauliPolynomial:
    """Return the repo-standard Jordan-Wigner annihilation operator."""

    return fermion_minus_operator(repr_mode="JW", nq=int(n_qubits), j=int(mode))


__all__ = [
    "PauliPolynomial",
    "PauliTerm",
    "fermion_minus_operator",
    "fermion_plus_operator",
    "jw_annihilation",
    "jw_creation",
]
