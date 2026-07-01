from pathlib import Path
import basis_set_exchange
from src.utilities.log import log
from src.set_param_object import p

special_basis = {"SBKJC": "SBKJC-VDZ"}

#
#   write basis set file
#

def write_basis_lib_file(basis_file, unique_elements, basis_map=None):
    """
    Write a Psi4-compatible BASIS library file for the given elements.
    """
    basis_file = Path(basis_file)
    # delete existing file if present
    if basis_file.exists():
        return
    basis_map = p.basis_set
    # iterate over elements
    for symbol in sorted(unique_elements):
        basis_name = basis_map.get(symbol)
        if basis_name is None:
            log.error(f"No basis set mapping found for element: {symbol}")
        elif basis_name in special_basis:
            basis_name = special_basis[basis_name]
        # retrieve basis string
        basis_str = basis_set_exchange.get_basis(
            basis_name,
            elements=symbol,
            fmt="psi4",
            header=False
        )
        if basis_str is None:
            log.error(f"Basis set '{basis_name}' not found for element {symbol}")
        # append to file
        with basis_file.open("a") as f:
            f.write(basis_str)
            f.write("\n")

#
#     basis set  ->  setup
#

def setup_basis_set(coord_file, basis_file_name):
    """
    Read a XYZ coordinate file and write a basis-set library
    for the unique atomic species found.
    """
    coord_file = Path(coord_file)
    if not coord_file.exists():
        log.error(f"Coordinate file not found: {coord_file}")
    elements = set()
    # open coord file
    with coord_file.open("r") as f:
        try:
            natoms = int(f.readline().strip())
        except ValueError:
            raise ValueError("First line of coordinate file must be the number of atoms")
        f.readline()  # skip comment / unit line
        for _ in range(natoms):
            line = f.readline().split()
            if not line:
                raise ValueError("Unexpected end of coordinate file")
            elements.add(line[0])
    write_basis_lib_file(basis_file_name, sorted(elements))