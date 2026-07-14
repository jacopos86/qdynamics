#!/usr/bin/env python
"""
main_psi4.py - Psi4 job script
------------------------------
Usage:
    python main_psi4.py h2.inp
"""
import argparse
import logging
from pathlib import Path

from src.parameters.set_param_object import p
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.chemistry.psi4.input_parser import parse_input
from src.chemistry.psi4.electron_solver import Psi4Driver, setup_basis_set
from src.chemistry.psi4 import read_write_psi4_results as out

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s: %(message)s')


def run_psi4_electronic_structure():
    """
    Run the Psi4 first-principles electronic-structure workflow.
    """
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t START PSI4 CALCULATION")
        log.info("\n")
        log.info("\t " + p.sep)
    # prepare/write basis set
    setup_basis_set(p.coordinate_file, p.basis_set_file)
    # set up psi4 driver
    psi4_obj = Psi4Driver(p.basis_set_file, p.psi4_calc_parameters)
    # -------------------------------------
    #  1)  geometry structure
    # -------------------------------------
    geometry = psi4_obj.psi4_geometry_driver(
        p.coordinate_file,
        p.optimized_coordinate_file,
        p.charge,
        p.multiplicity
    )
    # -------------------------------------
    #  2)  electronic structure
    # -------------------------------------
    WF = psi4_obj.psi4_elec_struct_driver(geometry)
    # -------------------------------------
    #    3)  model initialization
    # -------------------------------------
    S_obj, MO_obj, DM_obj, He = psi4_obj.set_electronic_operators(WF)
    # run tests
    psi4_obj.run_consistency_tests(WF, S_obj, MO_obj, DM_obj)
    # finalize calculation -> perform MO basis conversion
    psi4_obj.build_operators_MO_basis(MO_obj, DM_obj, He)
    # run energy tests
    psi4_obj.energy_report(He, WF, DM_obj)
    # if required compute electron vibration coupling
    psi4_obj.set_elecvibr_inter(WF)
    exit()


def PSI4_elec_gs_driver():
    """
    Backward-compatible name for the Psi4 electronic-structure workflow.
    """
    return run_psi4_electronic_structure()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Job input file (.inp)")
    args = parser.parse_args()

    cfg = parse_input(args.input)
    work_dir = args.input.resolve().parent

    setup_basis_set(
        cfg["coordinate_file"],
        cfg["basis_set_file"],
        basis_map=cfg["basis_set"],
        work_dir=work_dir,
    )
    psi4_obj = Psi4Driver(
        cfg["basis_set_file"],
        cfg["psi4_calc_parameters"],
        work_dir=work_dir,
    )
    geometry = psi4_obj.psi4_geometry_driver(
        cfg["coordinate_file"],
        cfg["optimized_coordinate_file"],
        cfg["charge"],
        cfg["multiplicity"],
    )
    wavefunction = psi4_obj.psi4_elec_struct_driver(geometry)
    overlap, mo, density_matrix, hamiltonian = psi4_obj.set_electronic_operators(
        wavefunction
    )
    psi4_obj.run_consistency_tests(wavefunction, overlap, mo, density_matrix)
    psi4_obj.build_operators_MO_basis(mo, density_matrix, hamiltonian)
    psi4_obj.energy_report(hamiltonian, wavefunction, density_matrix)

    prefix = cfg["output"]
    if prefix is None:
        prefix = args.input.with_suffix("")
    elif not prefix.is_absolute():
        prefix = work_dir / prefix

    meta = {
        "mol_str": Path(work_dir / cfg["coordinate_file"]).read_text(),
        "basis": str(cfg["basis_set"]),
        "method": cfg["psi4_calc_parameters"]["method"],
        "charge": cfg["charge"],
        "spin": cfg["multiplicity"] - 1,
        "unit": cfg["unit"],
        "xc": "",
        "e_scf": float(wavefunction.energy.magnitude),
        "converged": True,
        "nelec": int(wavefunction.mol_struct.nel),
        "nmo": int(wavefunction.nmo()),
        "ms2": int(cfg["multiplicity"] - 1),
    }
    if cfg["write_h5"]:
        out.write_wavefunction_h5(prefix.with_suffix(".h5"), wavefunction, meta)
    if cfg["write_matrix_elements"]:
        out.write_matrix_elements_h5(
            prefix.with_name(prefix.name + "_matrix.h5"),
            hamiltonian.hij,
            hamiltonian.Iijkl,
            hamiltonian.Vnn.magnitude,
            meta,
        )
    log.info(f"Job done. E_SCF = {wavefunction.energy.magnitude:.10f} Ha")


if __name__ == "__main__":
    main()
