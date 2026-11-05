import logging
import numpy as np
from pyscf.hessian import thermo

"""
vibration_solver.py
-------------------
Molecular vibrational analysis module: computation only.

Responsibility of this module:
  - Compute the nuclear Hessian on a converged SCF solution
  - Mass-weight and diagonalize -> normal modes and harmonic frequencies
"""

log = logging.getLogger(__name__)


class VibrationalSolver:
    """
    Harmonic vibrational analysis for molecules.

    Takes a converged PySCFDriver, computes the analytic nuclear Hessian
    and extracts harmonic frequencies, normal modes and reduced masses.
    Translations/rotations are projected out (3N-6 / 3N-5 modes).
    """

    def __init__(self, driver, exclude_trans=True, exclude_rot=True):
        """
        Parameters
        ----------
        driver : PySCFDriver
            Driver with a converged SCF solution (run_scf() already called).
        exclude_trans, exclude_rot : bool
            Project out translational / rotational modes (default True).
        """
        driver._check_ready()
        if not driver._converged:
            log.warning("SCF is not converged - Hessian may be unreliable.")
        self.driver = driver
        self.exclude_trans = exclude_trans
        self.exclude_rot = exclude_rot
        self.hessian = None
        self.results = None

    # -------------------------------------------------
    # computation
    # -------------------------------------------------
    def run(self):
        """
        Compute Hessian and perform harmonic analysis.

        Returns
        -------
        dict with keys:
          freq_au         : harmonic frequencies omega (a.u.)
          freq_wavenumber : harmonic frequencies (cm^-1), imaginary
                            modes appear as complex entries
          norm_mode       : (nmodes, natm, 3) mass-deweighted normal modes
          reduced_mass    : (nmodes,) in AMU
          force_const_dyne: (nmodes,) force constants (dyne/cm)
        """
        mf = self.driver.mf
        log.info("Computing analytic nuclear Hessian ...")
        self.hessian = mf.Hessian().kernel()
        self.results = thermo.harmonic_analysis(
            self.driver.mol, self.hessian,
            exclude_trans=self.exclude_trans,
            exclude_rot=self.exclude_rot,
        )
        n_imag = int(np.sum(self.results['freq_au'] < 0))
        if n_imag > 0:
            log.warning(f"{n_imag} imaginary frequencies found - "
                        "geometry is not a minimum.")
        freqs = self.results['freq_wavenumber']
        log.info(f"Vibrational analysis: {len(freqs)} modes, "
                 f"frequencies (cm^-1): {np.array2string(freqs.real, precision=1)}")
        return self.results

    def _check_ready(self):
        if self.results is None:
            raise RuntimeError("Vibrational analysis has not been run yet. "
                               "Call run() first.")

    def get_frequencies(self):
        """Harmonic frequencies in cm^-1 (may contain imaginary modes)."""
        self._check_ready()
        return self.results['freq_wavenumber']

    def get_normal_modes(self):
        """Mass-deweighted normal mode displacements, (nmodes, natm, 3)."""
        self._check_ready()
        return self.results['norm_mode']

    def get_reduced_masses(self):
        """Reduced masses in AMU, (nmodes,)."""
        self._check_ready()
        return self.results['reduced_mass']
