import numpy as np
from src.utilities.log import log

#
#   Molecular orbitals class
#

class molecular_orbitals_class:
    """
    Molecular orbitals class
    """
    def __init__(self):
        self.C = None
        # up / down
        self.Ca= None
        self.Cb= None
        # n. MOs
        self.nmo = None
    def set_mo_from_WF_obj(self, WF):
        """
        Set molecular orbitals from a Psi4 Wavefunction object.
        Stores AO->MO coefficient matrices as NumPy arrays.
        """
        if WF is None:
            log.error("WF object is None")
        # n. MOs
        self.nmo = WF.nmo()
        self.Ca = np.asarray(WF.Ca())
        self.Cb = np.asarray(WF.Cb())
        if self.Ca.shape[1] != self.nmo:
            log.error(
                f"Inconsistent alpha MO shape: {self.Ca.shape}, expected nmo={self.nmo}"
            )
        if self.Cb.shape[1] != self.nmo:
            log.error(
                f"Inconsistent beta MO shape: {self.Cb.shape}, expected nmo={self.nmo}"
            )
        # closed-shell case: Ca == Cb
        if np.allclose(self.Ca, self.Cb):
            self.C = self.Ca.copy()
        else:
            self.C = None
    # check orthogonality
    def _check_orthogonality(self, S):
        """
        Check MO orthogonality:
            C^T S C = I

        Parameters
        ----------
        S : ndarray
            AO overlap matrix
        reference : str
            Psi4 reference: 'rhf', 'rohf', 'uhf', 'rks', 'uks'
        tol : float
            Numerical tolerance
        """
        S = np.asarray(S)
        if self.C is not None:
            self._check_matrix(self.C, S, label="Restricted MOs")
        else:
            if self.Ca is None or self.Cb is None:
                log.error("MO coefficients not initialized")
            self._check_matrix(self.Ca, S, label="Alpha MOs")
            self._check_matrix(self.Cb, S, label="Beta MOs")
    def _check_matrix(self, C, S, label, tol=1.e-8):
        M = C.T @ S @ C
        I = np.eye(M.shape[0])
        err = np.max(np.abs(M - I))
        if err > tol:
            log.error(f"{label}: C^T S C != I  (max error = {err:.3e})")
        else:
            log.info(f"\t {label}: C^T S C = I  (max error = {err:.3e})")
    def run_tests(self, S):
        self._check_orthogonality(S)