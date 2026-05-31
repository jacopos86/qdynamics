import numpy as np
import psi4
from pathlib import Path
from src.parameters.set_param_object import p
from src.utilities.log import log
from src.io_module.write_xyz_file import write_xyz
from src.common.units import Q_

#
#   This module performs geometry optimization
#
#   define the molecule structure
#

class Psi4Molecule:
    """
    PSI4 molecular class
    """
    def __init__(self, xyz_file, charge, multiplicity):
        self.xyz_file = Path(xyz_file)
        self.xyz_file = Path(p.work_dir) / self.xyz_file
        self.charge = charge
        self.multiplicity = multiplicity
        # geometry
        self.geometry = self._load_geometry()
        self.nel = self._compute_electrons()
        self.print_info_data()
    def _load_geometry(self):
        """
        load geometry in PSI4:
        remove symmetry line -> no symmetry used for now
        """
        with open(self.xyz_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 3:
            log.error(f"XYZ file too short: {self.xyz_file}")
        # first line: number of atoms
        try:
            nat = int(lines[0])
        except ValueError:
            log.error(f"First line must be number of atoms, got: {lines[0]}")
        # second line: units/comment
        unit_line = lines[1].lower()
        if unit_line in ["angstrom", "ang", "a"]:
            psi4_units = "angstrom"
        elif unit_line in ["bohr", "au", "a.u."]:
            psi4_units = "bohr"
        else:
            log.error(
                f"Unsupported unit line '{lines[1]}' in {self.xyz_file}. "
                "Use 'Angstrom' or 'Bohr'."
            )
        # remaining lines: atoms
        atom_lines = lines[2:]
        if len(atom_lines) != nat:
            log.error(
                f"XYZ atom count mismatch: header says {nat}, found {len(atom_lines)} atom lines"
            )
        geom = f"{self.charge} {self.multiplicity}\n"
        geom += "\n".join(atom_lines) + "\n"
        geom += f"units {psi4_units}\n"
        geom += f"symmetry c1\n"
        log.info("\t " + p.sep)
        log.info("\t GEOM SENT TO PSI4:")
        log.info("\n"+geom)
        return psi4.geometry(geom)
    def _compute_electrons(self):
        Z = sum(self.geometry.Z(i) for i in range(self.geometry.natom()))
        return int(Z - self.charge)
    def compute_nuclear_repulsion_energy(self):
        return Q_(self.geometry.nuclear_repulsion_energy(), "hartree")
    def print_info_data(self):
        log.info(f"\t Total number atoms: {self.geometry.natom()}")
        log.info(f"\t Total num. electrons: {self.nel}")
        log.info(f"\t Multiplicity: {self.geometry.multiplicity()}")
        log.info(f"\t System charge: {self.geometry.molecular_charge()}")
        log.info(f"\t Repulsion energy: {self.compute_nuclear_repulsion_energy()}")

#
#    geometry optimization routine
#

def geometry_optimization(coordinate_file, optimized_coordinate_file, charge, multiplicity, method):
    '''
    geometry optimization interface
    '''
    atom_struct = Psi4Molecule(coordinate_file, charge, multiplicity)
    # set optimization run
    E_SCF, wfn = psi4.optimize(
        method.lower(), 
        molecule=atom_struct.geometry, 
        return_wfn=True
    )
    # save data on file
    write_xyz(atom_struct, optimized_coordinate_file, p.work_dir)
    atom_struct.print_info_data()
    return atom_struct