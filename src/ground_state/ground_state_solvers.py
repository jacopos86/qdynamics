# This is the main subroutine
# for non Markovian calculation of electronic systems
#
import numpy as np
from src.set_param_object import p
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.backends.psi4.BasisSet_module import setup_basis_set
from src.backends.psi4.run import Psi4Driver

# ====================================================
#
#     Hubbard model solver
#
# ====================================================

def solve_Hubbard_model():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t COMPUTE GROUND-STATE HUBBARD MODEL")
        log.info("\n")
        log.info("\t " + p.sep)

# ====================================================
#
#     PSI4 solver
#
# ====================================================

def PSI4_elec_gs_driver():
    '''
    use PSI4 data to perform RT dynamics
    '''
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t START PSI4 CALCULATION")
        log.info("\n")
        log.info("\t " + p.sep)
    exit()
    # prepare/write basis set
    setup_basis_set(p.coordinate_file, p.basis_set_file)
    # set up psi4 driver
    psi4_obj = Psi4Driver()
    # set up system object
    # ystem.init_atomic_structure()
    # optimize system geometry
    E_SCF, wfn = psi4_obj.psi4_geometry_driver()