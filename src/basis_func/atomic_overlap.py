import numpy as np
from src.utilities.log import log
from src.parameters.set_param_object import p

class AO_overlap_class:
    """
    AO overlap matrix and its first derivatives
    """
    def __init__(self, toler=1.0e-6):
        # overlap matrix
        self.S = None
        self.gradS = None
        self.TOLER = toler
        # num. basis func.
        self.nbf = None
        self.natom = None
    def build_from_WF(self, WF, with_gradient=True):
        """
        Build overlap matrix and optionally its nuclear derivatives
        from a Wavefunction class.
        """
        mints = WF.mints()
        self.natom = WF.mol_struct.geometry.natom()
        self.nbf = WF.wfn.basisset().nbf()
        # set overlap matrix
        self.set_overlap_matr(mints)
        # set atomic gradients
        if with_gradient:
            self.set_overlap_matr_grad(mints)
        # info
        self.print_info()
    def set_overlap_matr(self, mints):
        """
        Read AO overlap matrix from Psi4 MintsHelper.
        """
        S = np.asarray(mints.ao_overlap())
        if S.shape != (S.shape[0], S.shape[0]):
            log.error(f"Overlap matrix is not square: shape={S.shape}")
        assert self.nbf == S.shape[0]
        self.S = S.copy()
    def set_overlap_matr_grad(self, mints):
        """
        Read first derivatives of AO overlap matrix:
            gradS[xyz, atom, mu, nu]
        """
        if self.nbf is None or self.natom is None:
            log.error("nbf/natoms not initialized. Call build_from_mints first.")
        self.gradS = np.zeros((3, self.natom, self.nbf, self.nbf))
        for ia in range(self.natom):
            gS = mints.ao_oei_deriv1("OVERLAP", ia)
            for idx in range(3):
                A = np.asarray(gS[idx])
                if A.shape != (self.nbf, self.nbf):
                    log.error(
                        f"Gradient overlap shape mismatch for atom {ia}, dir {idx}: "
                        f"{A.shape} != ({self.nbf}, {self.nbf})"
                    )
                self.gradS[idx, ia, :, :] = A
    def _check_overlap_matr_properties(self):
        """
        Check:
         - diagonal close to 1
         - symmetry
         - no element larger than 1 by more than tolerance
        """
        if self.S is None:
            log.error("Overlap matrix S is not initialized")
        # diagonal
        diag_err = np.max(np.abs(np.diag(self.S) - 1.0))
        # symmetry
        sym_err = np.max(np.abs(self.S - self.S.T))
        # optional sanity check
        max_elem = np.max(np.abs(self.S))
        if diag_err > self.TOLER:
            log.error(f"Overlap diagonal deviates from 1 (max error = {diag_err:.3e})")
        if sym_err > self.TOLER:
            log.error(f"Overlap matrix is not symmetric (max error = {sym_err:.3e})")
        if max_elem > 1.0 + self.TOLER:
            log.error(f"Overlap matrix contains element > 1 (max abs element = {max_elem:.6f})")
        log.info(
            f"\t AO overlap matrix checks passed "
            f"(diag err = {diag_err:.3e}, sym err = {sym_err:.3e})"
        )
    def print_info(self):
        if self.S is None:
            log.warning("AO overlap matrix not initialized")
            return
        log.info(f"\t nbf: {self.nbf}")
        log.info(f"\t S shape: {self.S.shape}")
        if self.gradS is not None:
            log.info(f"\t gradS shape: {self.gradS.shape}")
        log.info("\t " + p.sep)
    def run_tests(self):
        self._check_overlap_matr_properties()