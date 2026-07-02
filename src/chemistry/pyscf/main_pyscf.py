#!/usr/bin/env python
"""
main.py - job script
--------------------
Usage:
    python main.py h2o.inp

See input_parser.py (or the bundled h2o.inp example) for the input
file format.
"""
import argparse
import logging
import sys
from pathlib import Path

from src.chemistry.pyscf.input_parser import parse_input
from src.chemistry.pyscf.pyscf_solver import PySCFDriver
from src.chemistry.pyscf.vibration_solver import VibrationalSolver
from src.chemistry.pyscf.electron_phonon_solver import ElectronPhononSolver
from src.chemistry.pyscf import read_write_pyscf_results as out

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s: %(message)s')
log = logging.getLogger('main')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', type=Path, help='Job input file (.inp)')
    args = p.parse_args()

    # ---------- stage 0: parse input ----------
    cfg = parse_input(args.input)

    # ---------- stage 1: compute ----------
    driver = PySCFDriver(
        cfg['mol_str'],
        basis=cfg['basis'],
        spin=cfg['spin'],
        charge=cfg['charge'],
        unit=cfg['unit'],
        method=cfg['method'],
        xc=cfg['xc'],
    )
    e_scf = driver.run_scf()

    if not driver._converged:
        log.error("SCF did not converge; refusing to write unreliable results.")
        sys.exit(1)

    h_mo, eri_mo_packed, Enuc = driver.get_mo_integrals(compact=True)

    # ---------- stage 2: write ----------
    prefix = cfg['output']
    meta = {
        'mol_str': cfg['mol_str'],
        'basis': cfg['basis'],
        'method': driver.method,
        'charge': cfg['charge'],
        'spin': cfg['spin'],
        'unit': cfg['unit'],
        'xc': cfg['xc'],
        'e_scf': float(e_scf),
        'converged': driver._converged,
        'nelec': int(driver.mol.nelectron),
        'nmo': int(h_mo.shape[0]),
        'ms2': int(driver.mol.spin),
    }

    if cfg['write_h5']:
        out.write_h5(prefix.with_suffix('.h5'),
                     driver.mf.mo_coeff, driver.mf.mo_energy, driver.mf.mo_occ,
                     meta)

    if cfg['write_fcidump']:
        out.write_fcidump(prefix.with_suffix('.fcidump'),
                          h_mo, eri_mo_packed, Enuc,
                          nmo=meta['nmo'], nelec=meta['nelec'],
                          ms2=meta['ms2'], tol=cfg['fcidump_tol'])

    # ---------- stage 3: vibrational analysis ----------
    if cfg['write_vibration']:
        vib = VibrationalSolver(driver)
        vib_results = vib.run()
        out.write_vibration_h5(prefix.with_name(prefix.name + '_vib.h5'),
                               vib_results, meta)

    # ---------- stage 4: electron-phonon coupling ----------
    if cfg['write_eph']:
        eph_solver = ElectronPhononSolver(driver)
        eph_mat, omega = eph_solver.run(mo_rep=True)
        out.write_eph_h5(prefix.with_name(prefix.name + '_eph.h5'),
                         eph_mat, omega,
                         {**meta, 'eph_basis': 'MO'})

    log.info(f"Job done. E_SCF = {e_scf:.10f} Ha")


if __name__ == '__main__':
    main()
