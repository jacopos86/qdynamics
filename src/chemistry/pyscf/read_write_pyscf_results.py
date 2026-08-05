"""
read_write_pyscf_results.py
-----------------
Output module for pyscf results

"""
import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)

def write_h5(path, mo_coeff, mo_energy, mo_occ, meta):
    """
    Write SCF orbital data + metadata to HDF5.

    Parameters
    ----------
    path : str or Path
    mo_coeff, mo_energy, mo_occ : SCF orbital data (needed later for
        active-space selection / integral rebuild without rerunning SCF)
    meta : dict
        Must contain: basis, method, charge, spin, unit, xc,
        e_scf, converged, nelec, nmo, ms2, mol_str
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        g = f.create_group('metadata')
        for k, v in meta.items():
            g.attrs[k] = v if v is not None else ''
        f.create_dataset('mo_coeff', data=mo_coeff, compression='gzip', chunks=True)
        f.create_dataset('mo_energy', data=mo_energy)
        f.create_dataset('mo_occ', data=mo_occ)

    log.info(f"[output] HDF5 written: {path}")
    return path


def write_matrix_elements_h5(path, h1, h2, nuclear_repulsion, meta):
    """Write full spin-orbital matrix elements using the shared schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        g = f.create_group('metadata')
        for k, v in meta.items():
            g.attrs[k] = v if v is not None else ''
        g.attrs['h1_convention'] = 'spin_orbital_interleaved'
        g.attrs['h2_convention'] = 'chemist_spin_orbital_coulomb'
        g.attrs['h2_antisymmetrized'] = False
        f.create_dataset('h1', data=np.asarray(h1), compression='gzip', chunks=True)
        f.create_dataset('h2', data=np.asarray(h2), compression='gzip', chunks=True)
        f.create_dataset('nuclear_repulsion', data=float(nuclear_repulsion))
    log.info(f"[output] matrix elements HDF5 written: {path}")
    return path


def write_vibration_h5(path, results, meta):
    """
    Write vibrational analysis results to HDF5.

    Parameters
    ----------
    path : str or Path
    results : dict
        Output of VibrationalSolver.run() (freq_au, freq_wavenumber,
        norm_mode, reduced_mass, force_const_dyne, ...)
    meta : dict
        Job metadata (basis, method, mol_str, ...)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        g = f.create_group('metadata')
        for k, v in meta.items():
            g.attrs[k] = v if v is not None else ''
        for key in ('freq_au', 'freq_wavenumber', 'norm_mode',
                    'reduced_mass', 'force_const_dyne'):
            if key in results:
                # imaginary frequencies -> complex arrays, h5py handles them
                f.create_dataset(key, data=np.asarray(results[key]))

    log.info(f"[output] vibrational HDF5 written: {path}")
    return path


def write_eph_h5(path, eph_mat, omega, meta):
    """
    Write electron-phonon coupling matrix elements to HDF5.

    Parameters
    ----------
    path : str or Path
    eph_mat : (nmodes, n, n) coupling matrix elements (a.u.)
    omega : (nmodes,) mode frequencies (a.u.)
    meta : dict
        Job metadata; should record the basis of eph_mat ('MO' or 'AO')
        under key 'eph_basis'.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, 'w') as f:
        g = f.create_group('metadata')
        for k, v in meta.items():
            g.attrs[k] = v if v is not None else ''
        f.create_dataset('eph_mat', data=np.asarray(eph_mat),
                         compression='gzip', chunks=True)
        f.create_dataset('omega', data=np.asarray(omega))

    log.info(f"[output] eph HDF5 written: {path} "
             f"(nmodes={len(omega)}, mat={np.shape(eph_mat)})")
    return path


def load_h5(path):
    """Read back an HDF5 file written by the writers above. Returns a plain
    dict with a 'metadata' entry plus every dataset in the file."""
    path = Path(path)
    out = {}
    with h5py.File(path, 'r') as f:
        out['metadata'] = dict(f['metadata'].attrs)
        for key in f.keys():
            if key != 'metadata':
                out[key] = f[key][()]
    log.info(f"[output] Loaded {path}")
    return out
