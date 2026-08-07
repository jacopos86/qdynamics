"""Finite-difference mean-field electron--phonon coupling."""
import logging
import numpy as np
from pyscf.eph import eph_fd

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
        self.eph_mat, self.omega = eph_fd.kernel(
            self.driver.mf,
            disp=self.fd_step,
            mo_rep=mo_rep,
            cutoff_frequency=self.cutoff_frequency,
            keep_imag_frequency=self.keep_imag_frequency,
        )
        return self.eph_mat, self.omega
