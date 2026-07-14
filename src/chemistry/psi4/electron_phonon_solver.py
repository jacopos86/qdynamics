import numpy as np


class ElectronPhononSolver:
    _KINETIC_LABEL = "KINETIC"
    _POTENTIAL_LABEL = "POTENTIAL"

    def __init__(self):
        # gradient one electron integrals
        # atomic orbitals
        self.grad_T_ao = None
        self.grad_V_ao = None
        self.grad_H0_ao = None
        # molecular orbital objects
        self.grad_hij = None

    # AO basis set
    def set_AO_operators(self, WF):
        self.set_ao_kinetic_operator(WF)
        self.set_ao_one_part_potential(WF)
        self.set_ao_one_part_hamiltonian()
        exit()
        self.set_ao_two_part_hamiltonian(WF)

    # set AO kinetic operator
    def set_ao_kinetic_operator(self, WF):
        nat = WF.mol_struct.geometry.natom()
        nbf = WF.wfn.basisset().nbf()
        mints = WF.mints()
        self.grad_T_ao = np.zeros((3, nat, nbf, nbf))
        for ia in range(nat):
            gT = mints.ao_oei_deriv1(self._KINETIC_LABEL, ia)
            for xyz in range(3):
                self.grad_T_ao[xyz, ia] = np.asarray(gT[xyz])

    # set AO one particle potential
    def set_ao_one_part_potential(self, WF):
        nat = WF.mol_struct.geometry.natom()
        nbf = WF.wfn.basisset().nbf()
        mints = WF.mints()
        self.grad_V_ao = np.zeros((3, nat, nbf, nbf))
        for ia in range(nat):
            gV = mints.ao_oei_deriv1(self._POTENTIAL_LABEL, ia)
            for xyz in range(3):
                self.grad_V_ao[xyz, ia] = np.asarray(gV[xyz])

    # set AO one particle hamiltonian
    def set_ao_one_part_hamiltonian(self):
        self.grad_H0_ao = self.grad_T_ao + self.grad_V_ao


ElecVibrCoupling = ElectronPhononSolver
