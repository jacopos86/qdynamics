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

from src.utilities.log import log
from src.chemistry.psi4.input_parser import parse_input
from src.chemistry.psi4.electron_solver import Psi4Driver, setup_basis_set
from src.chemistry.psi4.vibration_solver import VibrationalSolver
from src.chemistry.psi4.electron_phonon_fd_solver import FiniteDifferenceElectronPhononSolver
from src.chemistry.psi4 import read_write_psi4_results as out

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(name)s: %(message)s')


def run_psi4_electronic_structure():
    """
    Run the Psi4 first-principles electronic-structure workflow.
    """
    from src.parallelization.mpi import mpi
    from src.parameters.set_param_object import p

    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t START PSI4 CALCULATION")
        log.info("\n")
        log.info("\t " + p.sep)
    # prepare/write basis set
    setup_basis_set(
        p.coordinate_file,
        p.basis_set_file,
        basis_map=p.basis_set,
        work_dir=p.work_dir,
    )
    # set up psi4 driver
    psi4_obj = Psi4Driver(
        p.basis_set_file,
        p.psi4_calc_parameters,
        work_dir=p.work_dir,
    )
    # -------------------------------------
    #  1)  geometry structure
    # -------------------------------------
    geometry = psi4_obj.psi4_geometry_driver(
        p.coordinate_file,
        p.optimized_coordinate_file,
        p.charge,
        p.multiplicity,
        optimize_geometry=getattr(p, "optimize_geometry", True),
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

    prefix = _prefix_from_global_params(p)
    meta = {
        "mol_str": Path(p.work_dir, p.coordinate_file).read_text(),
        "basis": str(p.basis_set),
        "method": p.psi4_calc_parameters["method"],
        "charge": p.charge,
        "spin": p.multiplicity - 1,
        "unit": "",
        "xc": "",
        "e_scf": float(WF.energy.magnitude),
        "converged": True,
        "nelec": int(WF.mol_struct.nel),
        "nmo": int(WF.nmo()),
        "ms2": int(p.multiplicity - 1),
    }
    if getattr(p, "write_vibration", False):
        vib = VibrationalSolver(
            WF,
            method=p.psi4_calc_parameters["method"],
        )
        vib_results = vib.run()
        out.write_vibration_h5(
            prefix.with_name(prefix.name + "_vib.h5"), vib_results, meta
        )
    if getattr(p, "write_eph", False):
        psi4_obj.set_elecvibr_inter(WF)
    return WF


def PSI4_elec_gs_driver():
    """
    Backward-compatible name for the Psi4 electronic-structure workflow.
    """
    return run_psi4_electronic_structure()


def _prefix_from_global_params(p):
    output_hdf5 = getattr(p, "output_hdf5", None)
    if output_hdf5:
        path = Path(output_hdf5)
        if path.suffix:
            return path.with_suffix("")
        return path

    write_dir = Path(getattr(p, "write_dir", None) or p.work_dir)
    coordinate_stem = Path(p.coordinate_file).stem
    return write_dir / coordinate_stem


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
        optimize_geometry=cfg["optimize_geometry"],
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
    if cfg["write_h5"] or cfg["write_matrix_elements"]:
        out.write_matrix_elements_h5(
            prefix.with_name("ele.h5"),
            hamiltonian.hij, hamiltonian.Iijkl,
            hamiltonian.Vnn.magnitude, meta,
        )
    if cfg["write_vibration"]:
        vib = VibrationalSolver(
            wavefunction,
            method=cfg["psi4_calc_parameters"]["method"],
        )
        vib_results = vib.run()
        out.write_vibration_h5(
            prefix.with_name(f"{prefix.name}_vib.h5"), vib_results, meta
        )
    if cfg["write_eph"]:
        if not cfg["write_vibration"]:
            vib = VibrationalSolver(
                wavefunction, method=cfg["psi4_calc_parameters"]["method"])
            vib_results = vib.run()
        # EPH must use the complete 3N frequency/mode set.  The public
        # vibration result is intentionally filtered to physical modes, but
        # Psi4 EPH should match PySCF's cutoff-only selection.
        eph_vib_results = vib_results
        if "raw_freq_wavenumber" in vib_results:
            eph_vib_results = dict(vib_results)
            for key in ("freq_au", "freq_wavenumber", "norm_mode",
                        "reduced_mass", "force_const_dyne"):
                raw_key = f"raw_{key}"
                if raw_key in vib_results:
                    eph_vib_results[key] = vib_results[raw_key]
        eph_solver = FiniteDifferenceElectronPhononSolver(
            wavefunction, cfg["psi4_calc_parameters"]["method"],
            fd_step=cfg["eph_fd_step"])
        eph_mat, omega = eph_solver.run(eph_vib_results)
        out.write_eph_h5(
            prefix.with_name(f"{prefix.name}_eph.h5"), eph_mat, omega,
            {**meta, "eph_basis": "MO", "eph_method": "finite_difference"})
    log.info(f"Job done. E_SCF = {wavefunction.energy.magnitude:.10f} Ha")


if __name__ == "__main__":
    main()
