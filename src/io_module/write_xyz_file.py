from pathlib import Path
from src.common.units import Q_, Ang

#
#   write XYZ file
#

def write_xyz(atom_struct, optimized_coordinate_file, work_dir):
    xyz_path = Path(work_dir) / optimized_coordinate_file
    natoms = atom_struct.geometry.natom()
    # open file
    with open(xyz_path, "w") as f:
        f.write(f"{natoms}\n")
        f.write("Angstrom\n")
        for i in range(natoms):
            symb = atom_struct.geometry.label(i).capitalize()
            # bohr → Angstrom
            R = Q_(
                [
                    atom_struct.geometry.x(i),
                    atom_struct.geometry.y(i),
                    atom_struct.geometry.z(i),
                ],
                "bohr",
            ).to(Ang).magnitude
            f.write(f"{symb:2s}  {R[0]:16.8f}  {R[1]:16.8f}  {R[2]:16.8f}\n")