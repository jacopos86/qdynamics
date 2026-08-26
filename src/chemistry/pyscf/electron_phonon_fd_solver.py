"""Finite-difference mean-field electron--phonon coupling."""
import logging
import numpy as np
from pyscf.eph import eph_fd

from src.chemistry.pyscf.electron_phonon_solver import (
    pyscf_eph_cutoff,
    standardize_pyscf_eph,
)

log = logging.getLogger(__name__)


class FiniteDifferenceElectronPhononSolver:
    """Compute dF/dQ by displaced, independently converged SCF calculations.

    ``fd_step`` is the Cartesian central-difference displacement in Bohr.
    PySCF handles the mass-weighted mode transformation and AO/Pulay terms.
    """
    def __init__(self, driver, fd_step=0.001, cutoff_frequency=80.0,
                 keep_imag_frequency=False):
        driver._check_ready()
        if not driver._converged:
            raise RuntimeError("Reference SCF must be converged for FD EPH")
        if np.ndim(driver.mf.mo_coeff) == 3:
            raise NotImplementedError("FD EPH currently supports RHF/RKS references only")
        self.driver = driver
        self.fd_step = float(fd_step)
        self.cutoff_frequency = cutoff_frequency
        self.keep_imag_frequency = keep_imag_frequency
        self.eph_mat = self.omega = None

    def run(self, mo_rep=True):
        if not mo_rep:
            raise NotImplementedError("FD EPH currently returns MO-basis matrices only")
        # Displaced potentials are divided by a small Cartesian step.  The
        # ordinary PySCF SCF defaults can therefore leave visible noise in
        # weak EPH modes (CH2O/STO-3G is a representative case).  Apply the
        # same tight thresholds to the reference and to the copies made by
        # eph_fd.run_mfs, and reconverge the reference from its current DM.
        mf = self.driver.mf
        mf.conv_tol = min(float(mf.conv_tol), 1.0e-12)
        if mf.conv_tol_grad is None:
            mf.conv_tol_grad = 1.0e-8
        else:
            mf.conv_tol_grad = min(float(mf.conv_tol_grad), 1.0e-8)
        mf.max_cycle = max(int(mf.max_cycle), 100)
        mf.kernel(dm0=mf.make_rdm1())
        if not mf.converged:
            raise RuntimeError("Tight reference SCF did not converge for FD EPH")
        self.eph_mat, self.omega = eph_fd.kernel(
            mf,
            disp=self.fd_step,
            mo_rep=mo_rep,
            cutoff_frequency=pyscf_eph_cutoff(self.cutoff_frequency),
            keep_imag_frequency=self.keep_imag_frequency,
        )
        self.eph_mat, self.omega = standardize_pyscf_eph(
            self.eph_mat, self.omega)
        return self.eph_mat, self.omega
