"""Full fixed-electron-number diagonalization interface."""

from itertools import combinations

import numpy as np


def _apply_annihilation(state, orbital):
    if not (state >> orbital) & 1:
        return None
    phase = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state ^ (1 << orbital), phase


def _apply_creation(state, orbital):
    if (state >> orbital) & 1:
        return None
    phase = -1 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1
    return state | (1 << orbital), phase


def diagonalize_matrix_elements(h1, h2, nelec, Enuc=0.0):
    """Diagonalize a fixed-electron-number Hamiltonian."""
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    nso = h1.shape[0]
    if h1.shape != (nso, nso) or h2.shape != (nso, nso, nso, nso):
        raise ValueError("h1/h2 have incompatible shapes")
    if not 0 <= int(nelec) <= nso:
        raise ValueError("nelec must be between 0 and the number of spin orbitals")

    basis = [sum(1 << p for p in occ) for occ in combinations(range(nso), int(nelec))]
    index = {state: i for i, state in enumerate(basis)}
    H = np.zeros((len(basis), len(basis)), dtype=float)

    for col, state in enumerate(basis):
        for p in range(nso):
            for q in range(nso):
                x = _apply_annihilation(state, q)
                if x is None:
                    continue
                state_q, sign_q = x
                y = _apply_creation(state_q, p)
                if y is not None:
                    state_p, sign_p = y
                    H[index[state_p], col] += h1[p, q] * sign_q * sign_p

        for p in range(nso):
            for q in range(nso):
                for r in range(nso):
                    for s in range(nso):
                        x = _apply_annihilation(state, q)
                        if x is None:
                            continue
                        state_q, sign = x
                        x = _apply_annihilation(state_q, s)
                        if x is None:
                            continue
                        state_s, sign2 = x
                        x = _apply_creation(state_s, r)
                        if x is None:
                            continue
                        state_r, sign3 = x
                        x = _apply_creation(state_r, p)
                        if x is None:
                            continue
                        state_p, sign4 = x
                        H[index[state_p], col] += 0.5 * h2[p, q, r, s] * sign * sign2 * sign3 * sign4

    H = (H + H.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return {
        "electronic_energy": float(eigenvalues[0]),
        "total_energy": float(eigenvalues[0] + Enuc),
        "eigenvalues": eigenvalues,
        "eigenvector": eigenvectors[:, 0],
        "hamiltonian": H,
        "basis": basis,
        "nelec": int(nelec),
    }


def run_matrix_element_ed(driver, nelec=None):
    """Convenience wrapper for a converged ``PySCFDriver``."""
    driver._check_ready()
    h1, h2, Enuc = driver.get_matrix_elements()
    if nelec is None:
        nelec = driver.mol.nelectron
    return diagonalize_matrix_elements(h1, h2, nelec, Enuc)

__all__ = ["diagonalize_matrix_elements", "run_matrix_element_ed"]
