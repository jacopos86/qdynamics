"""
read_write_pyscf_results.py
-----------------
Output module for pyscf results

"""
import logging
from pathlib import Path

import h5py
import numpy as np
from pyscf.tools import fcidump

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


def write_fcidump(path, h_mo, eri_mo_packed, Enuc, nmo, nelec, ms2=0, tol=1e-12):
    """
    Write a standard FCIDUMP file (8-fold permutation symmetry).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fcidump.from_integrals(
        str(path),
        h1e=h_mo,
        h2e=eri_mo_packed,
        nmo=int(nmo),
        nelec=int(nelec),
        nuc=float(Enuc),
        ms=int(ms2),
        tol=tol,
    )
    log.info(f"[output] FCIDUMP written: {path} (nmo={nmo}, nelec={nelec})")
    return path


def load_h5(path):
    """Read back an HDF5 file written by write_h5(). Returns a plain dict."""
    path = Path(path)
    out = {}
    with h5py.File(path, 'r') as f:
        out['metadata'] = dict(f['metadata'].attrs)
        for key in ('mo_coeff', 'mo_energy', 'mo_occ'):
            out[key] = f[key][()]
    log.info(f"[output] Loaded {path}")
    return out
