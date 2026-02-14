from pathlib import Path
import numpy as np
from pyscf import gto, scf, dft

# -------------------------------------------------
#
#     pySCF driver module
#
# -------------------------------------------------

class PySCFDriver:
    """
    Generic PySCF SCF driver for molecules.
    Extracts SCF energy, 1- and 2-electron integrals.
    """

    def __init__(self, mol_str, basis='sto-3g', spin=0, charge=0, unit='Angstrom', method='RHF', xc=None):
        """
        Initialize the molecule and SCF method.
        
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
            Coordinates unit ('Angstrom' or 'Bohr')
        method : str
            'RHF', 'UHF', 'ROHF', 'KHF', 'KRHF', 'KUKS', 'UKS'
        xc : str or None
            Exchange-correlation functional for DFT
        """
        self.mol_str = mol_str
        self.basis = basis
        self.spin = spin
        self.charge = charge
        self.unit = unit
        self.method = method
        self.xc = xc

        self.mol = self._build_molecule()
        self.mf = self._build_scf_method()

    def _build_molecule(self):
        """Build PySCF molecule object"""
        mol = gto.M(
            atom=self.mol_str,
            basis=self.basis,
            spin=self.spin,
            charge=self.charge,
            unit=self.unit,
            verbose=0
        )
        log.info(f"Molecule built with {mol.natm} atoms, basis={self.basis}")
        return mol

    def _build_scf_method(self):
        """Initialize SCF object"""
        method = self.method.upper()
        if method in ['RHF', 'ROHF', 'UHF']:
            mf = getattr(scf, method)(self.mol)
        elif method in ['UKS', 'RKS']:
            mf = dft.RKS(self.mol) if method == 'RKS' else dft.UKS(self.mol)
            if self.xc is not None:
                mf.xc = self.xc
        else:
            raise ValueError(f"Unsupported SCF method: {method}")
        log.info(f"SCF method initialized: {method}")
        return mf

    def run_scf(self):
        """Run SCF and return total energy"""
        e_scf = self.mf.kernel()
        log.info(f"SCF energy: {e_scf} Ha")
        return e_scf

    def get_integrals(self):
        """Return 1- and 2-electron integrals in AO basis"""
        mol = self.mol
        mf = self.mf

        # Overlap, kinetic, nuclear potential
        S = mol.intor('int1e_ovlp')
        T = mol.intor('int1e_kin')
        V = mol.intor('int1e_nuc')
        H = T + V

        # 2-electron integrals
        eri = mol.intor('int2e')  # (μν|λσ)

        # Nuclear repulsion
        Enuc = mol.energy_nuc()

        log.info(f"Integrals extracted: {S.shape}, {H.shape}, {eri.shape}")
        return S, H, eri, Enuc

    def get_density_matrix(self):
        """Return AO density matrix"""
        if not hasattr(self.mf, 'mo_coeff'):
            raise RuntimeError("SCF has not been run yet")
        mo_coeff = self.mf.mo_coeff
        occ = self.mf.mo_occ
        D = mo_coeff @ np.diag(occ) @ mo_coeff.T
        return D