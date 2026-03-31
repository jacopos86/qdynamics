import numpy as np
import psi4
from src.common.units import Q_

#
#   Electronic hamiltonian class
#

class ElectronicHamiltonian:
    """
    Electronic Hamiltonian class
    """
    def __init__(self):
        # ACTIVE space objects
        self.H1p = None
        self.H2p = None
        # Vnn
        self.Vnn = None
        # TEI -> MO basis
        self.Iijkl = None
        # OEI -> MO basis
        self.hij = None
        # AO basis
        self.T_ao = None
        self.V_ao = None
        self.H0_ao = None
        self.Hee_ao = None
    # set nuclear interaction energy
    def set_nuclear_repulsion_energy(self, geometry):
        self.Vnn = Q_(
            geometry.nuclear_repulsion_energy(), 
            "hartree"
        )
    # AO basis set
    def set_AO_operators(self, mints):
        self.set_ao_kinetic_operator(mints)
        self.set_ao_one_part_potential(mints)
        self.set_ao_one_part_hamiltonian()
        self.set_ao_two_part_hamiltonian(mints)
    # set AO kinetic operator
    def set_ao_kinetic_operator(self, mints):
        self.T_ao = np.asarray(mints.ao_kinetic())
    # set AO one particle potential
    def set_ao_one_part_potential(self, mints):
        self.V_ao = np.asarray(mints.ao_potential())
    # set AO one particle hamiltonian
    def set_ao_one_part_hamiltonian(self):
        self.H0_ao = self.T_ao + self.V_ao
    # set AO one particle hamiltonian
    def set_ao_two_part_hamiltonian(self, mints):
        self.Hee_ao = np.asarray(mints.ao_eri())
    # AE single particle matr. elements
    def set_ae_1p_matr_elements(self, MO_obj):
        """
        Transform one-particle AO Hamiltonian to MO basis.

        Restricted case:
         hij[0] = C^T H_ao C
         hij[1] = hij[0]

        Unrestricted / open-shell case:
         hij[0] = Ca^T H_ao Ca
         hij[1] = Cb^T H_ao Cb
        """
        if self.H0_ao is None:
            log.error("AO one-particle Hamiltonian H0_ao is not initialized")
        _hij = [None, None]
        # Restricted case
        if MO_obj.C is not None:
            C = np.asarray(MO_obj.C)
            _hij[0] = C.T @ self.H0_ao @ C
            _hij[1] = _hij[0].copy()
        # Unrestricted / ROHF case
        else:
            if MO_obj.Ca is None or MO_obj.Cb is None:
                log.error("MO coefficients are not initialized")
            Ca = np.asarray(MO_obj.Ca)
            Cb = np.asarray(MO_obj.Cb)
            _hij[0] = Ca.T @ self.H0_ao @ Ca
            _hij[1] = Cb.T @ self.H0_ao @ Cb
        # set full spin orbital matrix
        self.hij = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo))
        for i in range(MO_obj.nmo):
            for j in range(MO_obj.nmo):
                self.hij[2*i, 2*j] = _hij[0][i,j]
                self.hij[2*i+1, 2*j+1] = _hij[1][i,j]
    def set_ae_2p_matr_elements(self, MO_obj):
        """
        General AO -> MO ERI transformation:
            (ij|kl) = C1 C2 C3 C4 (mu nu|la si)

        Uses chemist notation.
        """
        self.Iijkl = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo, 2*MO_obj.nmo, 2*MO_obj.nmo))
        self.Iijkl = np.einsum(
            "mi,nj,pk,ql,mnpq->ijkl",

        )
    def check_total_energy(self, WF, DM_obj):
        _tot_energy = WF.energy_1p.magnitude + WF.energy_2p.magnitude + self.Vnn.magnitude
        psi4.compare_values(WF.energy.magnitude, _tot_energy, 6, 'total energy')
        self._check_1p_energy(WF, DM_obj)
        self._check_2p_energy(WF, DM_obj)
    def _check_1p_energy(self, WF, DM_obj):
        D = DM_obj.Dae
        _en_1p = np.trace(D @ self.hij)
        en_1p = Q_(_en_1p, "hartree")
        psi4.compare_values(en_1p.magnitude, WF.energy_1p.magnitude, 6, 'one particle energy')
    def _check_2p_energy(self, WF, DM_obj):
        pass