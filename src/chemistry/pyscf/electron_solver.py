from pathlib import Path
import logging
import numpy as np
from pyscf import gto, scf, dft, ao2mo

"""
electron_solver.py
------------------
PySCF electronic-structure solver module: computation only.

Responsibility of this module:
  - Build molecule, run SCF
  - Extract AO / MO / spin-orbital integrals

NOT the responsibility of this module:
  - Writing any files -> see output_writers.py
  - Job workflow / orchestration -> see main.py
"""

log = logging.getLogger(__name__)

class PySCFDriver:
    """
    Generic PySCF SCF driver for molecules.
    Extracts SCF energy, 1- and 2-electron integrals (AO, MO, spin-orbital).
    """

    def __init__(self, mol_str, basis='sto-3g', spin=0, charge=0,
                 unit='Angstrom', method='RHF', xc=None):
        """
        Parameters
        ----------
        mol_str : str
            XYZ format coordinates as string
        basis : str
            Basis set name
        spin : int
            2S (number of unpaired electrons)
        charge : int
            Molecular charge
        unit : str
            'Angstrom' or 'Bohr'
        method : str
            'RHF', 'UHF', 'ROHF', 'RKS', 'UKS'
        xc : str or None
            Exchange-correlation functional for DFT
        """
        self.mol_str = mol_str
        self.basis = basis
        self.spin = spin
        self.charge = charge
        self.unit = unit
        self.method = method.upper()
        self.xc = xc
        self.mol = self._build_molecule()
        self.mf = self._build_scf_method()
        self._converged = False

    # -------------------------------------------------
    # setup
    # -------------------------------------------------
    def _build_molecule(self):
        mol = gto.M(
            atom=self.mol_str,
            basis=self.basis,
            spin=self.spin,
            charge=self.charge,
            unit=self.unit,
            verbose=0,
        )
        log.info(f"Molecule built with {mol.natm} atoms, basis={self.basis}")
        return mol

    def _build_scf_method(self):
        method = self.method
        if method in ('RHF', 'ROHF', 'UHF'):
            mf = getattr(scf, method)(self.mol)
        elif method in ('RKS', 'UKS'):
            mf = dft.RKS(self.mol) if method == 'RKS' else dft.UKS(self.mol)
            if self.xc is not None:
                mf.xc = self.xc
        else:
            raise ValueError(f"Unsupported SCF method: {method}")
        log.info(f"SCF method initialized: {method}")
        return mf

    # -------------------------------------------------
    # computation
    # -------------------------------------------------
    def run_scf(self):
        """Run SCF and return total energy."""
        e_scf = self.mf.kernel()
        self._converged = bool(self.mf.converged)
        if not self._converged:
            log.warning("SCF did NOT converge - downstream integrals may be unreliable.")
        log.info(f"SCF energy: {e_scf} Ha (converged={self._converged})")
        return e_scf

    def _check_ready(self):
        if not hasattr(self.mf, 'mo_coeff') or self.mf.mo_coeff is None:
            raise RuntimeError("SCF has not been run yet. Call run_scf() first.")

    def get_integrals(self):
        """AO basis: S, H(=T+V), ERI (chemist's notation), Enuc."""
        mol = self.mol
        S = mol.intor('int1e_ovlp')
        H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        eri = mol.intor('int2e')
        Enuc = mol.energy_nuc()
        return S, H, eri, Enuc

    def get_density_matrix(self):
        """AO density matrix. (Da, Db) for unrestricted, D for restricted."""
        self._check_ready()
        mo_coeff = self.mf.mo_coeff
        occ = self.mf.mo_occ
        if np.ndim(occ) == 2:  # UHF/UKS
            Da = mo_coeff[0] @ np.diag(occ[0]) @ mo_coeff[0].T
            Db = mo_coeff[1] @ np.diag(occ[1]) @ mo_coeff[1].T
            return Da, Db
        return mo_coeff @ np.diag(occ) @ mo_coeff.T

    def get_mo_integrals(self, compact=True):
        """
        MO basis integrals (restricted reference).

        Parameters
        ----------
        compact : bool
            If True (default), eri_mo is returned in PySCF's 8-fold
            symmetry-packed 2D format (~n^4/8 elements) - preferred
            for storage. If False, full (n,n,n,n) tensor.

        Returns
        -------
        h_mo, eri_mo, Enuc
        """
        self._check_ready()
        if np.ndim(self.mf.mo_occ) == 2:
            raise NotImplementedError(
                "get_mo_integrals() currently supports restricted references only."
            )
        mol = self.mol
        mo = self.mf.mo_coeff
        nmo = mo.shape[1]
        h_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        h_mo = mo.T @ h_ao @ mo
        eri_mo = ao2mo.kernel(mol, mo, compact=compact)
        if not compact:
            eri_mo = eri_mo.reshape(nmo, nmo, nmo, nmo)
        Enuc = mol.energy_nuc()
        log.info(f"MO integrals: h{h_mo.shape}, eri{eri_mo.shape} (compact={compact})")
        return h_mo, eri_mo, Enuc

    def get_spin_integrals(self):
        """
        Spin-orbital integrals for second quantization (restricted reference).
        Interleaved ordering (a0, b0, a1, b1, ...).

        NOTE: builds the full (2n)^4 tensor in memory - fine for small
        molecules, do NOT use for large basis sets. For storage, save
        MO integrals instead and expand on the fly at the consumer side.

        Returns
        -------
        h1_spin : (2n, 2n)
        h2_spin : (2n, 2n, 2n, 2n), antisymmetrized physicist's notation
        Enuc : float
        """
        h_mo, eri_mo, Enuc = self.get_mo_integrals(compact=False)
        nmo = h_mo.shape[0]
        nso = 2 * nmo

        h1_spin = np.zeros((nso, nso))
        h1_spin[0::2, 0::2] = h_mo   # alpha block
        h1_spin[1::2, 1::2] = h_mo   # beta block

        # chemist's (pq|rs) -> physicist's <pq|rs> = (pr|qs)
        eri_phys = eri_mo.transpose(0, 2, 1, 3)

        h2_spin = np.zeros((nso, nso, nso, nso))
        for sp in range(2):          # spin of p and r (must match)
            for sq in range(2):      # spin of q and s (must match)
                h2_spin[sp::2, sq::2, sp::2, sq::2] = eri_phys

        h2_spin = h2_spin - h2_spin.transpose(0, 1, 3, 2)  # <pq||rs>
        log.info(f"Spin-orbital integrals: h1{h1_spin.shape}, h2{h2_spin.shape}")
        return h1_spin, h2_spin, Enuc

    def get_matrix_elements(self):
        """
        Spin-orbital matrix elements in the shared first-principles format.

        Returns
        -------
        h1 : (2*nmo, 2*nmo)
            One-electron matrix elements in interleaved spin-orbital order.
        h2 : (2*nmo, 2*nmo, 2*nmo, 2*nmo)
            Two-electron Coulomb integrals in chemist notation (ij|kl), not
            antisymmetrized. This matches the Psi4 matrix-element writer.
        Enuc : float
            Nuclear repulsion energy.
        """
        h_mo, eri_mo, Enuc = self.get_mo_integrals(compact=False)
        nmo = h_mo.shape[0]
        h1 = np.zeros((2*nmo, 2*nmo))
        h1[0::2, 0::2] = h_mo
        h1[1::2, 1::2] = h_mo

        h2 = np.zeros((2*nmo, 2*nmo, 2*nmo, 2*nmo))
        for spin_ij in range(2):
            for spin_kl in range(2):
                h2[spin_ij::2, spin_ij::2, spin_kl::2, spin_kl::2] = eri_mo

        log.info(f"Matrix elements: h1{h1.shape}, h2{h2.shape}")
        return h1, h2, Enuc
