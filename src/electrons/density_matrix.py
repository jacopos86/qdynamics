import numpy as np
import psi4
from src.utilities.log import log

class DensityMatrix:
    """
    Density matrix container.

    Stores:
     - AO density matrices: Dao = [Da, Db]
      - MO occupation vectors
      - optional active-space / full-space diagonal MO density matrices
    """
    def __init__(self):
        # AO density matrices
        self.Dao = None          # [Da, Db]
        # MO-space density matrices
        self.Dact = None         # [Dact_a, Dact_b]
        self.Dae = None          # [Dae_a, Dae_b]
        # occupations
        self.nocc = np.zeros(2, dtype=int)
        self.nel = None
        self.f = None            # restricted spin-orbital occupations
        self.fa = None           # alpha spin-orbital occupations
        self.fb = None           # beta  spin-orbital occupations
    def set_occup_from_wfn(self, WF):
        """
        Set occupied orbital counts from Psi4 wavefunction.
        """
        self.nocc[0] = WF.nalpha()
        self.nocc[1] = WF.nbeta()
        self.nel = sum(self.nocc)
    def set_orbital_occup(self, MO_obj):
        """
        Build spin-orbital occupation arrays.
        """
        nmo = MO_obj.nmo
        if MO_obj.C is not None:
            self.f = np.zeros(2 * nmo)
            for i in range(self.nocc[0]):
                self.f[2 * i] = 1.0
                self.f[2 * i + 1] = 1.0
        elif MO_obj.Ca is not None or MO_obj.Cb is not None:
            self.fa = np.zeros(2 * nmo)
            self.fb = np.zeros(2 * nmo)
            for i in range(self.nocc[0]):
                self.fa[2 * i] = 1.0
            for i in range(self.nocc[1]):
                self.fb[2 * i + 1] = 1.0
        else:
            log.error("MO_obj is not initialized")
    def compute_dm_from_mo(self, MO_obj):
        """
        Build AO density matrices from MO coefficients.
        Dao = [Da, Db]
        """
        # restricted
        if MO_obj.C is not None:
            C = np.asarray(MO_obj.C)
            Da = np.einsum("mi,ni->mn", C[:,:self.nocc[0]], C[:,:self.nocc[0]])
            self.Dao = [Da, Da.copy()]
        # unrestricted / ROHF
        else:
            if MO_obj.Ca is None or MO_obj.Cb is None:
                log.error("MO coefficients not initialized")
            Ca = np.asarray(MO_obj.Ca)
            Cb = np.asarray(MO_obj.Cb)
            Da = np.einsum("mi,ni->mn", Ca[:,:self.nocc[0]], Ca[:,:self.nocc[0]])
            Db = np.einsum("mi,ni->mn", Cb[:,:self.nocc[1]], Cb[:,:self.nocc[1]])
            self.Dao = [Da, Db]
    def set_ae_space_dm(self, nmo):
        """
        Build full-space diagonal density matrix in spin-orbital MO basis.
        """
        self.Dae = np.zeros((2*nmo, 2*nmo))
        if self.f is not None:
            for i in range(self.f.shape[0]):
                self.Dae[i, i] = self.f[i]
        else:
            for i in range(self.fa.shape[0]):
                self.Dae[i, i] = self.fa[i]
                self.Dae[i, i] += self.fb[i]
    def _compare_total_electron_number(self, S, nel_expected, tol=1.0e-6):
        """
        Check Tr(S D) = number of electrons.
        """
        if self.Dao is None:
            log.error("AO density matrix not initialized")
        S = np.asarray(S)
        nel_up = np.trace(S @ self.Dao[0])
        nel_dw = np.trace(S @ self.Dao[1])
        nel = nel_up + nel_dw
        log.info(f"\t Tr(S Da) = {nel_up}")
        log.info(f"\t Tr(S Db) = {nel_dw}")
        err = abs(nel - nel_expected)
        log.info(f"\t Total electrons from density = {nel}")
        log.info(f"\t Expected electrons           = {nel_expected}")
        if err > tol:
            log.error(f"Electron number check failed (error = {err:.3e})")
        psi4.compare_values(nel, nel_expected, 4, 'number of electrons')
    def run_tests(self, S, nel_expected):
        self._compare_total_electron_number(S, nel_expected)