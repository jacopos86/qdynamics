import logging

import numpy as np

import psi4

"""
vibration_solver.py
-------------------
Molecular vibrational analysis for the Psi4.
"""

log = logging.getLogger(__name__)

HARTREE_TO_WAVENUMBER = 219474.6313705
AMU_TO_ELECTRON_MASS = 1822.888486209


class VibrationalSolver:
    """
    Harmonic vibrational analysis for molecules through Psi4.

    Parameters
    ----------
    wavefunction : Psi4Wavefunction
        Wrapper returned by Psi4Driver.psi4_elec_struct_driver().
    method : str, optional
        Psi4 method name. If omitted, the method used by the existing
        wavefunction is inferred where possible.
    exclude_trans, exclude_rot : bool
        Kept for API compatibility with the PySCF solver. Psi4's frequency
        driver performs the standard vibrational analysis; the fallback
        diagonalizer drops the lowest 3/5/6 near-zero modes.
    """

    def __init__(self, wavefunction, method=None, exclude_trans=True, exclude_rot=True):
        if wavefunction.wfn is None:
            raise RuntimeError("Psi4 wavefunction is not available.")
        self.wavefunction = wavefunction
        self.method = method or self._infer_method()
        self.exclude_trans = exclude_trans
        self.exclude_rot = exclude_rot
        self.hessian = None
        self.frequency_wfn = None
        self.results = None

    def run(self):
        """
        Run Psi4 frequency analysis and return vibrational data.

        Returns
        -------
        dict
            Keys follow the PySCF vibration solver where possible:
            freq_au, freq_wavenumber, norm_mode, reduced_mass,
            force_const_dyne. The raw Hessian is also included.
        """
        mol = self.wavefunction.mol_struct.geometry
        log.info("Computing Psi4 harmonic vibrational analysis ...")
        _, self.frequency_wfn = psi4.frequency(
            self.method,
            molecule=mol,
            return_wfn=True,
        )
        self.hessian = self._read_hessian(self.frequency_wfn)
        self.results = self._from_psi4_vibration_data(self.frequency_wfn)
        if self.results is None:
            log.warning(
                "Psi4 frequency metadata was not available; "
                "falling back to local Hessian diagonalization."
            )
            self.results = self._from_hessian_fallback(mol, self.hessian)

        self.results["hessian"] = self.hessian
        freqs = np.asarray(self.results["freq_wavenumber"])
        n_imag = int(np.sum(np.real(freqs) < 0.0))
        if n_imag > 0:
            log.warning(
                f"{n_imag} imaginary frequencies found - geometry is not a minimum."
            )
        log.info(
            f"Psi4 vibrational analysis: {len(freqs)} modes, "
            f"frequencies (cm^-1): {np.array2string(np.real(freqs), precision=1)}"
        )
        return self.results

    def _infer_method(self):
        try:
            return str(self.wavefunction.wfn.name()).lower()
        except Exception:
            return "scf"

    def _read_hessian(self, wfn):
        for attr in ("hessian", "Hessian"):
            try:
                value = getattr(wfn, attr)()
                return np.asarray(value)
            except Exception:
                pass
        raise RuntimeError(
            "Psi4 did not return a Hessian on the frequency wavefunction."
        )

    def _from_psi4_vibration_data(self, wfn):
        vibinfo = None
        for attr in ("frequency_analysis", "vibinfo"):
            try:
                candidate = getattr(wfn, attr)
                vibinfo = candidate() if callable(candidate) else candidate
                if vibinfo:
                    break
            except Exception:
                pass
        if not vibinfo:
            return None

        freq_cm = self._first_array(
            vibinfo,
            "freq_wavenumber",
            "frequency",
            "frequencies",
            "omega",
            "omega_cm",
        )
        if freq_cm is None:
            return None
        freq_cm = self._as_array(freq_cm)

        freq_au = self._first_array(vibinfo, "freq_au", "omega_au")
        if freq_au is None:
            freq_au = freq_cm / HARTREE_TO_WAVENUMBER
        else:
            freq_au = self._as_array(freq_au)

        norm_mode = self._first_array(
            vibinfo,
            "norm_mode",
            "normal_modes",
            "q",
            "x",
        )
        if norm_mode is not None:
            norm_mode = self._reshape_modes(self._as_array(norm_mode))

        reduced_mass = self._first_array(
            vibinfo,
            "reduced_mass",
            "mu",
            "mass",
        )
        force_const = self._first_array(
            vibinfo,
            "force_const_dyne",
            "k",
            "force_constant",
        )

        return {
            "freq_au": np.asarray(freq_au),
            "freq_wavenumber": freq_cm,
            "norm_mode": norm_mode if norm_mode is not None else np.empty((0, 0, 3)),
            "reduced_mass": (
                self._as_array(reduced_mass)
                if reduced_mass is not None
                else np.full(len(freq_cm), np.nan)
            ),
            "force_const_dyne": (
                self._as_array(force_const)
                if force_const is not None
                else np.full(len(freq_cm), np.nan)
            ),
        }

    def _first_array(self, mapping, *keys):
        for key in keys:
            try:
                if key in mapping:
                    value = mapping[key]
                    if isinstance(value, dict):
                        for subkey in ("data", "value", "values"):
                            if subkey in value:
                                return value[subkey]
                    return value
            except Exception:
                pass
        return None

    def _as_array(self, value):
        for attr in ("data", "value", "values"):
            if hasattr(value, attr):
                candidate = getattr(value, attr)
                value = candidate() if callable(candidate) else candidate
                break
        return np.asarray(value)

    def _reshape_modes(self, modes):
        modes = np.asarray(modes)
        nat = self.wavefunction.mol_struct.geometry.natom()
        if modes.ndim == 2 and modes.shape[0] == 3 * nat:
            modes = modes.T
        if modes.ndim == 2 and modes.shape[1] == 3 * nat:
            return modes.reshape(modes.shape[0], nat, 3)
        if modes.ndim == 3 and modes.shape[-2:] == (nat, 3):
            return modes
        return modes

    def _from_hessian_fallback(self, mol, hessian):
        nat = mol.natom()
        hessian = np.asarray(hessian).reshape(3 * nat, 3 * nat)
        masses_amu = np.asarray([mol.mass(i) for i in range(nat)], dtype=float)
        masses = np.repeat(masses_amu * AMU_TO_ELECTRON_MASS, 3)
        mass_weight = hessian / np.sqrt(np.outer(masses, masses))
        eigvals, eigvecs = np.linalg.eigh(mass_weight)

        omega = np.sign(eigvals) * np.sqrt(np.abs(eigvals))
        order = np.argsort(np.abs(omega))
        omega = omega[order]
        eigvecs = eigvecs[:, order]

        nremove = 0
        if self.exclude_trans:
            nremove += 3
        if self.exclude_rot:
            nremove += 2 if self._is_linear(mol) else 3
        keep = np.arange(nremove, len(omega))
        omega = omega[keep]
        eigvecs = eigvecs[:, keep]

        norm_mode = (eigvecs.T / np.sqrt(masses)).reshape(len(omega), nat, 3)
        freq_cm = omega * HARTREE_TO_WAVENUMBER
        return {
            "freq_au": omega,
            "freq_wavenumber": freq_cm,
            "norm_mode": norm_mode,
            "reduced_mass": np.full(len(omega), np.nan),
            "force_const_dyne": np.full(len(omega), np.nan),
        }

    def _is_linear(self, mol, tol=1.0e-6):
        nat = mol.natom()
        if nat <= 2:
            return True
        coords = np.asarray([[mol.x(i), mol.y(i), mol.z(i)] for i in range(nat)])
        coords -= coords.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(coords, full_matrices=False)
        return np.count_nonzero(singular_values > tol) <= 1

    def _check_ready(self):
        if self.results is None:
            raise RuntimeError(
                "Vibrational analysis has not been run yet. Call run() first."
            )

    def get_frequencies(self):
        """Harmonic frequencies in cm^-1."""
        self._check_ready()
        return self.results["freq_wavenumber"]

    def get_normal_modes(self):
        """Mass-deweighted normal mode displacements, (nmodes, natm, 3)."""
        self._check_ready()
        return self.results["norm_mode"]

    def get_reduced_masses(self):
        """Reduced masses in AMU where reported by Psi4."""
        self._check_ready()
        return self.results["reduced_mass"]
