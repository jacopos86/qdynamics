import psi4
from src.utilities.log import log
from src.backends.psi4.molecular_structure import Psi4Molecule

class Psi4Wavefunction:
    """
    Wrapper around psi4.core.Wavefunction.

    Supports:
      - building an empty wavefunction container (for MintsHelper / AO integrals)
      - running an actual electronic-structure calculation and storing the converged wfn
    """
    def __init__(self, mol_struct: Psi4Molecule):
        self.mol_struct = mol_struct
        # energy / wfn
        self.wfn = None
        self.energy = None
        self.energy_1p = None
        self.energy_2p = None
        self._build_empty()
    # build empty wfn
    def _build_empty(self):
        """
        Build an empty wavefunction container.
        Useful for MintsHelper / AO integrals before running SCF.
        """
        basis_name = psi4.core.get_global_option("BASIS")
        self.wfn = psi4.core.Wavefunction.build(self.mol_struct.geometry, basis_name)
        # check wfn
        if self.wfn.nirrep() != 1:
            raise ValueError(
                f"Wavefunction has nirrep={self.wfn.nirrep()}, expected 1. "
                "Use symmetry c1 in the geometry."
            )
    def mints(self):
        """
        Return MintsHelper from the current wavefunction.
        """
        if self.wfn is None:
            log.error("Wavefunction not available. Call build_empty() or compute().")
        return psi4.core.MintsHelper(self.wfn.basisset())
    def nalpha(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nalpha()
    def nbeta(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nbeta()
    def nirrep(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nirrep()
    def nmo(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nmo()
    def Ca(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.Ca()
    def Cb(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.Cb()
    def print_summary(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        log.info("\t Psi4 wavefunction summary")
        log.info(f"\t nirrep      : {self.wfn.nirrep()}")
        log.info(f"\t nalpha      : {self.wfn.nalpha()}")
        log.info(f"\t nbeta       : {self.wfn.nbeta()}")
        log.info(f"\t nmo         : {self.wfn.nmo()}")
        if self.energy is not None:
            log.info(f"\t energy      : {self.energy}")
        if self.energy_1p is not None:
            log.info(f"\t energy one-particle      : {self.energy_1p}")
        if self.energy_2p is not None:
            log.info(f"\t energy two-particle      : {self.energy_2p}")