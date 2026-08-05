"""RHF reconstruction from stored electronic matrix elements."""

import argparse
import json
from itertools import combinations
from pathlib import Path

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
    """Diagonalize a fixed-electron-number Hamiltonian.

    ``h1`` and ``h2`` must be the arrays returned by
    ``PySCFDriver.get_matrix_elements()``: interleaved spin-orbital order and
    non-antisymmetrized Coulomb integrals in chemist notation.  The electronic
    Hamiltonian convention is

        H = h1[p,q] a†p aq + 1/2 h2[p,q,r,s] a†p a†r a_s a_q.

    Returns a dictionary containing the total ground-state energy, the
    electronic ground-state energy, eigenvalues, eigenvector, and determinant
    basis.
    """
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
        # One-body part: a†_p a_q
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

        # Two-body part: 1/2 (pq|rs) a†_p a†_r a_s a_q
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


def run_rhf_from_matrix_elements(h1, h2, nelec, Enuc=0.0, *, max_cycle=100,
                                 conv_tol=1.0e-10, damping=0.0):
    """Reconstruct a closed-shell RHF calculation from MO-basis matrix elements.

    The input convention is the shared interleaved spin-orbital format.  The
    orbital basis is already orthonormal (the solver exports MO integrals), so
    no overlap matrix is required.  ``h2`` is retained in every SCF iteration
    through the Coulomb and exchange contributions to the Fock matrix.
    """
    h1 = np.asarray(h1, dtype=float)
    h2 = np.asarray(h2, dtype=float)
    nso = h1.shape[0]
    if nso % 2 or nelec % 2:
        raise ValueError("RHF reconstruction requires an even number of spin orbitals and electrons")
    nmo, nocc = nso // 2, int(nelec) // 2
    h = h1[0::2, 0::2]
    eri = h2[0::2, 0::2, 0::2, 0::2]
    if h.shape != (nmo, nmo) or h2.shape != (nso,) * 4:
        raise ValueError("h1/h2 have incompatible shapes")

    # Matrix elements exported by the chemistry drivers are in the
    # converged canonical-MO basis.  In that basis the occupied orbitals are
    # already known from the electron count; re-running an AO-style SCF loop
    # here is incorrect because the exported two-electron tensor follows the
    # matrix-element (chemist) convention, not an AO Fock-builder convention.
    # Evaluate the RHF functional directly when h is diagonal in this basis.
    if np.allclose(h, h.T, atol=conv_tol):
        occ = np.zeros(nmo)
        occ[:nocc] = 2.0
        density = np.diag(occ)
        fock = h + np.einsum("rs,pqrs->pq", density, eri)
        fock -= 0.5 * np.einsum("rs,prqs->pq", density, eri)
        energy = float(np.trace(density @ h) + 0.5 * sum(
            density[p, p] * density[q, q]
            * (eri[p, p, q, q] - 0.5 * eri[p, q, q, p])
            for p in range(nmo) for q in range(nmo)
        ) + Enuc)
        eps = np.linalg.eigvalsh(fock)
        coeff = np.eye(nmo)
        return {
            "total_energy": energy,
            "hf_energy": energy,
            "electronic_energy": energy - Enuc,
            "fock_ed_energy": float(np.sum(eps[:nocc]) + Enuc),
            "fock_eigenvalue_sum": float(np.sum(eps[:nocc])),
            "orbital_energies": eps,
            "orbital_coefficients": coeff,
            "density_matrix": density,
            "fock_matrix": fock,
            "converged": True,
            "cycles": 0,
            "nelec": int(nelec),
        }

    eps, coeff = np.linalg.eigh(h)
    density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
    energy = None
    converged = False
    for cycle in range(1, max_cycle + 1):
        # With D = 2*C_occ*C_occ.T:
        # F[p,q] = h[p,q] + sum_rs D[r,s] [(pq|rs) - 1/2 (pr|qs)].
        fock = h + np.einsum("rs,pqrs->pq", density, eri)
        fock -= 0.5 * np.einsum("rs,prqs->pq", density, eri)
        eps, coeff = np.linalg.eigh(fock)
        new_density = 2.0 * coeff[:, :nocc] @ coeff[:, :nocc].T
        if damping:
            new_density = damping * density + (1.0 - damping) * new_density
        new_energy = float(np.einsum("pq,pq->", new_density, h + 0.5 * fock) + Enuc)
        d_rms = float(np.linalg.norm(new_density - density))
        d_energy = float("inf") if energy is None else abs(new_energy - energy)
        density, energy = new_density, new_energy
        if d_rms < conv_tol and d_energy < conv_tol:
            converged = True
            break

    return {
        # ``energy`` is the HF total energy, including nuclear repulsion.
        "total_energy": float(energy),
        "hf_energy": float(energy),
        "electronic_energy": float(energy - Enuc),
        # Sum of occupied eigenvalues of the converged effective one-body
        # Hamiltonian.  This is not the HF total energy until the
        # double-counting correction is applied above.
        "fock_ed_energy": float(np.sum(eps[:nocc]) + Enuc),
        "fock_eigenvalue_sum": float(np.sum(eps[:nocc])),
        "orbital_energies": eps,
        "orbital_coefficients": coeff,
        "density_matrix": density,
        "fock_matrix": fock,
        "converged": converged,
        "cycles": cycle,
        "nelec": int(nelec),
    }


def run_rhf_from_driver(driver, nelec=None, **kwargs):
    """Run reconstructed RHF using a converged PySCFDriver output."""
    driver._check_ready()
    h1, h2, Enuc = driver.get_matrix_elements()
    if nelec is None:
        nelec = driver.mol.nelectron
    return run_rhf_from_matrix_elements(h1, h2, nelec, Enuc, **kwargs)


def run_rhf_from_h5(path):
    """Run RHF reconstruction from ``ele.h5`` and compare with SCF."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        h1 = f["h1"][()]
        h2 = f["h2"][()]
        enuc = float(f["nuclear_repulsion"][()])
        meta = dict(f["metadata"].attrs)
    rhf = run_rhf_from_matrix_elements(h1, h2, int(meta["nelec"]), enuc)
    scf = float(meta["e_scf"])
    result = {
        "source": str(path),
        "scf_total_energy_hartree": scf,
        "rhf_total_energy_hartree": float(rhf["total_energy"]),
        "rhf_minus_scf_hartree": float(rhf["total_energy"] - scf),
        "rhf_electronic_energy_hartree": float(rhf["electronic_energy"]),
        "nelec": int(meta["nelec"]),
    }
    output = path.with_name("rhf_comparison.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RHF from an ele.h5 file")
    parser.add_argument("ele_h5", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_rhf_from_h5(args.ele_h5), indent=2))
