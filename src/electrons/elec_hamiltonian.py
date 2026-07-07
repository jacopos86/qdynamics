import numpy as np
import psi4
from src.common.units import Q_
from src.utilities.log import log

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
        Transform AO electron-repulsion integrals to spin-orbital MO basis.

        Uses chemist notation:
            (ij|kl) =
                integral dr1 dr2
                phi_i(r1) phi_j(r1) (1 / r12) phi_k(r2) phi_l(r2)

            (ij|kl) = C_mu,i C_nu,j C_lambda,k C_sigma,l
                      (mu nu|lambda sigma)

        In spin-orbital form:
            <p q | r s> =
                integral dx1 dx2
                psi_p(x1) psi_q(x1) (1 / r12) psi_r(x2) psi_s(x2)

            <p q | r s> =
                (ij|kl) delta_spin(p,q) delta_spin(r,s)

        Spin-orbital convention:
            spatial MO i alpha -> 2*i
            spatial MO i beta  -> 2*i + 1

        The Coulomb operator is spin independent, so spin-orbital ERIs are
        nonzero only when spin(i) == spin(j) and spin(k) == spin(l).
        """
        if self.Hee_ao is None:
            log.error("AO two-particle Hamiltonian Hee_ao is not initialized")
        # internal num. molecular orbitals
        nmo = MO_obj.nmo
        Iao = np.asarray(self.Hee_ao)
        coeffs = [np.asarray(MO_obj.Ca), np.asarray(MO_obj.Cb)]
        # I_ijkl molecular orbital basis
        self.Iijkl = np.zeros((2*nmo, 2*nmo, 2*nmo, 2*nmo))
        # loop over spin blocks
        for spin_ij in range(2):
            Cij = coeffs[spin_ij]
            for spin_kl in range(2):
                Ckl = coeffs[spin_kl]
                block = np.einsum(
                    "mi,nj,pk,ql,mnpq->ijkl",
                    Cij,
                    Cij,
                    Ckl,
                    Ckl,
                    Iao,
                    optimize=True,
                )
                i_idx = slice(spin_ij, 2*nmo, 2)
                j_idx = slice(spin_ij, 2*nmo, 2)
                k_idx = slice(spin_kl, 2*nmo, 2)
                l_idx = slice(spin_kl, 2*nmo, 2)
                self.Iijkl[i_idx, j_idx, k_idx, l_idx] = block
        self.H2p = self.Iijkl
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
        occ = np.diag(DM_obj.Dae)
        en_2p = 0.0
        for p in range(occ.shape[0]):
            if occ[p] == 0.0:
                continue
            for q in range(occ.shape[0]):
                if occ[q] == 0.0:
                    continue
                en_2p += 0.5 * occ[p] * occ[q] * (
                    self.Iijkl[p, p, q, q] - self.Iijkl[p, q, q, p]
                )
        psi4.compare_values(en_2p, WF.energy_2p.magnitude, 6, 'two particle energy')