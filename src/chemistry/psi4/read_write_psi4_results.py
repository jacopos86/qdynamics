"""
read_write_psi4_results.py
--------------------------
Output module for Psi4 results.

The primary HDF5 schema intentionally matches
src.chemistry.pyscf.read_write_pyscf_results:

  metadata   HDF5 group attrs
  mo_coeff   dataset
  mo_energy  dataset
  mo_occ     dataset
"""
import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


def _write_metadata(h5_file, meta):
    group = h5_file.create_group("metadata")
    for key, value in meta.items():
        group.attrs[key] = value if value is not None else ""


def write_h5(path, mo_coeff, mo_energy, mo_occ, meta):
    """
    Write SCF orbital data + metadata to HDF5.

    This matches the PySCF writer schema so downstream readers can use one
    code path for PySCF and Psi4 results.

    Parameters
    ----------
    path : str or Path
    mo_coeff, mo_energy, mo_occ : SCF orbital data
    meta : dict
        Job metadata. Use the same keys as the PySCF writer where possible:
        basis, method, charge, spin, unit, xc, e_scf, converged, nelec, nmo,
        ms2, mol_str.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        _write_metadata(f, meta)
        f.create_dataset("mo_coeff", data=np.asarray(mo_coeff),
                         compression="gzip", chunks=True)
        f.create_dataset("mo_energy", data=np.asarray(mo_energy))
        f.create_dataset("mo_occ", data=np.asarray(mo_occ))

    log.info(f"[output] Psi4 HDF5 written: {path}")
    return path


def _psi4_array(value):
    return np.asarray(value)


def orbital_data_from_wavefunction(wavefunction):
    """
    Convert a Psi4Wavefunction wrapper into PySCF-style orbital arrays.

    Restricted wavefunctions return:
      mo_coeff  shape (nao, nmo)
      mo_energy shape (nmo,)
      mo_occ    shape (nmo,)

    Unrestricted wavefunctions return spin-stacked arrays with shape
    (2, nao, nmo), (2, nmo), and (2, nmo), matching PySCF UHF convention.
    """
    wfn = wavefunction.wfn
    ca = _psi4_array(wfn.Ca())
    cb = _psi4_array(wfn.Cb())
    ea = _psi4_array(wfn.epsilon_a())
    eb = _psi4_array(wfn.epsilon_b())
    nmo = int(wfn.nmo())
    nalpha = int(wfn.nalpha())
    nbeta = int(wfn.nbeta())

    occ_a = np.zeros(nmo)
    occ_b = np.zeros(nmo)
    occ_a[:nalpha] = 1.0
    occ_b[:nbeta] = 1.0

    if np.allclose(ca, cb) and np.allclose(ea, eb):
        mo_coeff = ca
        mo_energy = ea
        mo_occ = occ_a + occ_b
    else:
        mo_coeff = np.asarray([ca, cb])
        mo_energy = np.asarray([ea, eb])
        mo_occ = np.asarray([occ_a, occ_b])

    return mo_coeff, mo_energy, mo_occ


def write_wavefunction_h5(path, wavefunction, meta):
    """
    Convenience wrapper that writes a Psi4Wavefunction using the shared schema.
    """
    mo_coeff, mo_energy, mo_occ = orbital_data_from_wavefunction(wavefunction)
    return write_h5(path, mo_coeff, mo_energy, mo_occ, meta)


def write_matrix_elements_h5(path, h1, h2, nuclear_repulsion, meta):
    """
    Write first-principles matrix elements to a backend-independent HDF5 file.

    Schema matches the PySCF writer:
      metadata
      h1
      h2
      nuclear_repulsion
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        _write_metadata(f, meta)
        f["metadata"].attrs["h1_convention"] = "spin_orbital_interleaved"
        f["metadata"].attrs["h2_convention"] = "chemist_spin_orbital_coulomb"
        f["metadata"].attrs["h2_antisymmetrized"] = False
        f.create_dataset("h1", data=np.asarray(h1),
                         compression="gzip", chunks=True)
        f.create_dataset("h2", data=np.asarray(h2),
                         compression="gzip", chunks=True)
        f.create_dataset("nuclear_repulsion", data=float(nuclear_repulsion))

    log.info(f"[output] Psi4 matrix elements HDF5 written: {path}")
    return path


def write_vibration_h5(path, results, meta):
    """
    Write vibrational analysis results to HDF5.

    Schema intentionally matches the PySCF vibration writer where possible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        _write_metadata(f, meta)
        for key in (
            "freq_au",
            "freq_wavenumber",
            "norm_mode",
            "reduced_mass",
            "force_const_dyne",
            "hessian",
        ):
            if key in results and results[key] is not None:
                f.create_dataset(key, data=np.asarray(results[key]))

    log.info(f"[output] Psi4 vibrational HDF5 written: {path}")
    return path


def write_eph_h5(path, eph_mat, omega, meta):
    """
    Write electron-phonon coupling matrix elements to HDF5.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        _write_metadata(f, meta)
        f.create_dataset("eph_mat", data=np.asarray(eph_mat),
                         compression="gzip", chunks=True)
        f.create_dataset("omega", data=np.asarray(omega))

    log.info(f"[output] Psi4 eph HDF5 written: {path}")
    return path


def load_h5(path):
    """Read back an HDF5 file written by the writers above."""
    path = Path(path)
    out = {}
    with h5py.File(path, "r") as f:
        out["metadata"] = dict(f["metadata"].attrs)
        for key in f.keys():
            if key != "metadata":
                out[key] = f[key][()]

    log.info(f"[output] Loaded {path}")
    return out
