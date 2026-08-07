"""Compiled Pauli-string action helpers (exyz convention).

These utilities implement fast statevector action for a single Pauli string
without forming dense matrices.

Math:
    P|psi>  via permutation + phase
    exp(-i * dt * c * P)|psi> = cos(theta)|psi> - i sin(theta) P|psi>
    where theta = dt * Re(c)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledPauliAction:
    label_exyz: str
    perm: np.ndarray
    phase: np.ndarray


def compile_pauli_action_exyz(label_exyz: str, nq: int) -> CompiledPauliAction:
    """Compile an exyz Pauli label into permutation/phase arrays.

    The label convention is q_(n-1)...q_0 (qubit 0 rightmost).
    """
    dim = 1 << int(nq)
    idx = np.arange(dim, dtype=np.int64)
    perm = idx.copy()
    phase = np.ones(dim, dtype=complex)

    for q in range(int(nq)):
        op = str(label_exyz)[int(nq) - 1 - q]
        bits = ((idx >> q) & 1).astype(np.int8)
        sign = (1 - 2 * bits).astype(np.int8)

        if op == "e":
            continue
        if op == "x":
            perm ^= (1 << q)
            continue
        if op == "y":
            perm ^= (1 << q)
            phase *= 1j * sign
            continue
        if op == "z":
            phase *= sign
            continue
        raise ValueError(f"Unsupported Pauli symbol '{op}' in '{label_exyz}'.")

    return CompiledPauliAction(label_exyz=str(label_exyz), perm=perm, phase=phase)


def apply_compiled_pauli(psi: np.ndarray, action: CompiledPauliAction) -> np.ndarray:
    """Apply a compiled Pauli action to a statevector."""
    out = np.empty_like(psi)
    out[action.perm] = action.phase * psi
    return out


def apply_exp_term(
    psi: np.ndarray,
    action: CompiledPauliAction,
    coeff: complex,
    dt: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """Apply exp(-i * dt * coeff * P) using the compiled Pauli action."""
    coeff_c = complex(coeff)
    if abs(coeff_c.imag) > float(tol):
        raise ValueError(f"Imaginary coefficient encountered for {action.label_exyz}: {coeff_c}")
    theta = float(dt) * float(coeff_c.real)
    ppsi = apply_compiled_pauli(psi, action)
    return math.cos(theta) * psi - 1j * math.sin(theta) * ppsi


__all__ = [
    "CompiledPauliAction",
    "compile_pauli_action_exyz",
    "apply_compiled_pauli",
    "apply_exp_term",
]

