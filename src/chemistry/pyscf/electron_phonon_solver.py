import logging
import numpy as np
from pyscf import eph
from pyscf.data.nist import AMU2AU, MP_ME


# PySCF EPH treats atom masses reported in amu as though multiplying by the
# proton/electron mass ratio converted them to electron masses.  A uniform
# mass rescaling leaves mode eigenvectors unchanged, so omega and the
# zero-point-normalized coupling can be converted back to the standard amu
# convention exactly.
EPH_FREQUENCY_SCALE = np.sqrt(MP_ME / AMU2AU)
EPH_COUPLING_SCALE = np.sqrt(EPH_FREQUENCY_SCALE)


def pyscf_eph_cutoff(cutoff_frequency):
    """Return the raw PySCF cutoff corresponding to an amu-scale cutoff."""
    return float(cutoff_frequency) / EPH_FREQUENCY_SCALE


def standardize_pyscf_eph(eph_mat, omega):
    """Convert raw PySCF EPH matrices and frequencies to the amu convention."""
    return (np.asarray(eph_mat) * EPH_COUPLING_SCALE,
            np.asarray(omega) * EPH_FREQUENCY_SCALE)

"""
electron_phonon_solver.py
-------------------------
Electron-phonon (vibronic) coupling module: computation only.

Responsibility of this module:
  - Compute first-order electron-phonon coupling matrix elements
    g_{I,pq} = <p| dV/dQ_I |q> / sqrt(2 w_I)   (a.u.)
    for each vibrational normal mode I
"""

log = logging.getLogger(__name__)


class ElectronPhononSolver:
    """
    Electron-phonon coupling for molecules via PySCF's analytic eph module.

    Takes a converged PySCFDriver and computes the coupling matrix
    elements between orbitals for each normal mode. Supports the same
    references as pyscf.eph: RHF, UHF, RKS, UKS.
    """

    def __init__(self, driver, cutoff_frequency=80.0, keep_imag_frequency=False):
        """
        Parameters
        ----------
        driver : PySCFDriver
            Driver with a converged SCF solution (run_scf() already called).
        cutoff_frequency : float
            Modes below this frequency (cm^-1) are discarded (default 80,
            removes residual translations/rotations).
        keep_imag_frequency : bool
            Keep imaginary-frequency modes (default False).
        """
        driver._check_ready()
        if not driver._converged:
            log.warning("SCF is not converged - eph couplings may be unreliable.")
        self.driver = driver
        self.cutoff_frequency = cutoff_frequency
        self.keep_imag_frequency = keep_imag_frequency
        self.eph_mat = None
        self.omega = None
        self.mo_rep = None

    # -------------------------------------------------
    # computation
    # -------------------------------------------------
    def run(self, mo_rep=True):
        """
        Compute electron-phonon coupling matrix elements.

        Parameters
        ----------
        mo_rep : bool
            If True (default) the couplings are returned in the MO basis
            (ready for second quantization); if False, in the AO basis.

        Returns
        -------
        eph_mat : (nmodes, n, n) ndarray
            Coupling matrix elements g_{I,pq} in a.u. (Hartree); the
            1/sqrt(2 w_I) zero-point factor is already included.
        omega : (nmodes,) ndarray
            Mode frequencies in a.u. (Hartree), same order as eph_mat.
        """
        mf = self.driver.mf
        log.info("Computing electron-phonon coupling matrix elements ...")
        myeph = eph.EPH(mf,
                        cutoff_frequency=pyscf_eph_cutoff(
                            self.cutoff_frequency),
                        keep_imag_frequency=self.keep_imag_frequency)
        self.eph_mat, self.omega = myeph.kernel(mo_rep=mo_rep)
        self.eph_mat, self.omega = standardize_pyscf_eph(
            self.eph_mat, self.omega)
        self.mo_rep = mo_rep
        basis = 'MO' if mo_rep else 'AO'
        log.info(f"eph couplings: {self.eph_mat.shape} ({basis} basis), "
                 f"{len(self.omega)} modes, "
                 f"omega (a.u.): {np.array2string(self.omega, precision=6)}")
        return self.eph_mat, self.omega

    def _check_ready(self):
        if self.eph_mat is None:
            raise RuntimeError("eph calculation has not been run yet. "
                               "Call run() first.")

    def get_coupling_matrix(self):
        """Coupling matrix elements, (nmodes, n, n) in a.u."""
        self._check_ready()
        return self.eph_mat

    def get_frequencies(self):
        """Mode frequencies in a.u., (nmodes,)."""
        self._check_ready()
        return self.omega
