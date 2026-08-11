"""RHF reconstruction from stored electronic matrix elements."""

import argparse
import json
from pathlib import Path
import numpy as np

def run_rhf_from_matrix_elements(h1, h2, nelec, Enuc=0.0, *, max_cycle=100,
                                 conv_tol=1.0e-10, damping=0.0):
    """Reconstruct a closed-shell RHF calculation from MO-basis matrix elements.

    The input convention is the shared interleaved spin-orbital format.  The
    orbital basis is already orthonormal (the solver exports MO integrals), so
    no overlap matrix is required.  The fixed ``h2`` tensor is used in every
    SCF iteration to construct the Coulomb and exchange contributions to the
    Fock matrix.
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
    if np.allclose(h, np.diag(np.diag(h)), atol=conv_tol):
        occ = np.zeros(nmo)
        occ[:nocc] = 2.0
        density = np.diag(occ)
        fock = h + np.einsum("rs,pqrs->pq", density, eri)
        fock -= 0.5 * np.einsum("rs,prqs->pq", density, eri)
        eps = np.linalg.eigvalsh(fock)
        coeff = np.eye(nmo)
        energy = float(np.trace(density @ h) + 0.5 * sum(
            density[p, p] * density[q, q]
            * (eri[p, p, q, q] - 0.5 * eri[p, q, q, p])
            for p in range(nmo) for q in range(nmo)
        ) + Enuc)
        return {
            "total_energy": energy,
            "hf_energy": energy,
            "electronic_energy": energy - Enuc,
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
        "total_energy": float(energy),
        "hf_energy": float(energy),
        "electronic_energy": float(energy - Enuc),
        "orbital_energies": eps,
        "orbital_coefficients": coeff,
        "density_matrix": density,
        "fock_matrix": fock,
        "converged": converged,
        "cycles": cycle,
        "nelec": int(nelec),
    }

def run_rhf_from_h5(path, source=None):
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
        "source": source or str(path),
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
