from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.utilities.input_parser import parser
from src.parameters.set_param_object import p
from src.chemistry.psi4.main_psi4 import run_psi4_electronic_structure
from src.hamiltonians.main_hubbard_holstein import run_hubbard_holstein_model
from src.drivers.spin_dyn_solvers import compute_spinsys_dephas

#
#  spin qubit driver
#

def spin_system_driver(yml_file):
    calc_type1 = parser.parse_args().ct1[0]
    calc_type2 = parser.parse_args().ct2
    # -----------------------------------
    #       ground state
    # ----------------------------------- 
    if calc_type1 == "GS":
        _run_spin_gs_driver(yml_file, calc_type2)
    # -----------------------------------
    #       real time evolution
    # ----------------------------------- 
    elif calc_type1 == "RT":
        _run_spin_rt_driver(yml_file, calc_type2)
    else:
        _print_usage()
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")

# ------------------------------------------------------------
#
#    GS spin solver
#
# ------------------------------------------------------------

def _run_spin_gs_driver(yml_file, calc_type2):
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t SPIN SYSTEM CALCULATION")
        log.info("\n")
    # spin homogeneous part
    if calc_type2 == "homo":
        ZFS_CALC = True
        HFI_CALC = False
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
            log.info("\n")
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t HOMOGENEOUS SPIN - DEPHASING")
            log.info("\t ZFS_CALC: " + str(ZFS_CALC))
            log.info("\t HFI_CALC: " + str(HFI_CALC))
            log.info("\n")
            log.info("\t " + p.sep)
    # spin inhomogeneous section
    elif calc_type2 == "inhomo":
        ZFS_CALC = False
        HFI_CALC = True
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
            log.info("\n")
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t INHOMOGENEOUS SPIN - DEPHASING")
            log.info("\t ZFS_CALC: " + str(ZFS_CALC))
            log.info("\t HFI_CALC: " + str(HFI_CALC))
            log.info("\n")
            log.info("\t " + p.sep)
    # full spin section
    elif calc_type2 == "full":
        ZFS_CALC = True
        HFI_CALC = True
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
            log.info("\n")
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t INHOMOGENEOUS SPIN - DEPHASING")
            log.info("\t ZFS_CALC: " + str(ZFS_CALC))
            log.info("\t HFI_CALC: " + str(HFI_CALC))
            log.info("\n")
            log.info("\t " + p.sep)
    else:
        _print_usage()
        log.error("\t calc_type2 wrong: " + str(calc_type2))
    # read yml file
    p.read_yml_data(yml_file)
    # calculation handler
    T2_calc_handler = compute_spinsys_ground_state(ZFS_CALC, HFI_CALC)
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t" + p.sep)
        log.info("\t PRINT DATA ON FILES")
        T2_calc_handler.print_results()
        log.info("\t" + p.sep)
        log.info("\n")
    mpi.comm.Barrier()

# --------------------------------------------------------------
# 
#    FULL CALC. -> REAL TIME
#
# --------------------------------------------------------------

def _run_spin_rt_driver(yml_file, calc_type2):
    if mpi.rank == mpi.root:
        log.info("\t T2 CALCULATION -> STARTING")
        log.info("\t REAL TIME DYNAMICS")
        log.info("\n")
        log.info("\t " + p.sep)
    # read yml file
    p.read_yml_data(yml_file)
    p.check_consistency()
    # homogeneous calculation
    if calc_type2 == "homo":
        ZFS_CALC = True
        HFI_CALC = False
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
            log.info("\n")
    # inhomogeneous
    elif calc_type2 == "inhomo":
        ZFS_CALC = False
        HFI_CALC = True
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
            log.info("\n")
    # full spin
    elif calc_type2 == "full":
        ZFS_CALC = True
        HFI_CALC = True
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
            log.info("\n")
    else:
        if mpi.rank == mpi.root:
            log.warning("\t REAL TIME DYNAMICS -> calc_type2 : homo/inhomo/full")
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")
    # calculation handler
    compute_spinsys_dephas(ZFS_CALC, HFI_CALC)
    mpi.comm.Barrier()

#
#    electron system calculation driver
#

def elec_system_driver(yml_file):
    calc_type1 = parser.parse_args().ct1[0]
    calc_type2 = parser.parse_args().ct2
    # ground state solver
    if calc_type1 == "GS":
        _run_elec_gs_driver(yml_file, calc_type2)
    # real-time solver
    elif calc_type1 == "RT":
        _run_elec_rt_driver(yml_file)
    else:
        _print_usage()
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")

# ------------------------------------------------------------
#
#    ELECTRONIC GROUND-STATE solver
#
# ------------------------------------------------------------

def _run_elec_gs_driver(yml_file, calc_type2):
    if mpi.rank == mpi.root:
        log.info("\t GROUND STATE CALCULATION -> STARTING")
        log.info("\n")
        log.info("\t " + p.sep)
    # read yaml file
    p.read_yml_data(yml_file)
    # solve HH model
    if calc_type2 == "HH_MODEL":
        calc_handler = run_hubbard_holstein_model()
    # PSI4 branch
    elif calc_type2 == "PSI4":
        _run_elec_psi4_quantum_pipeline()
    # PYSCF branch
    elif calc_type2 == "PYSCF":
        if mpi.rank == mpi.root:
            log.warning("\t PYSCF GROUND STATE DRIVER NOT IMPLEMENTED YET")
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")
    else:
        if mpi.rank == mpi.root:
            log.warning("\t GROUND STATE SOLVER")
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")

# --------------------------------------------------------------
# 
#    ELECTRONIC CALC. -> REAL TIME
#
# --------------------------------------------------------------

def _run_elec_rt_driver(yml_file):
    if mpi.rank == mpi.root:
        log.info("\t ELECTRON DYNAMICS CALCULATION -> STARTING")
        log.info("\t REAL TIME DYNAMICS")
        log.info("\n")
        log.info("\t " + p.sep)
    # read yaml file
    p.read_yml_data(yml_file)
    # run electron driver
    calc_handler = elec_dyn_driver()

# --------------------------------------------------------------
# 
#    HOLSTEIN-HUBBARD ELECTRONIC CALC.
#
# --------------------------------------------------------------

def _run_hh_model_pipeline():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t USING HOLSTEIN-HUBBARD MODEL DATA")
        log.info("\t START HH MODEL PIPELINE")
        log.info("\n")
    #
    # temporary first step:
    # preserve current behavior exactly
    #
    calc_handler = run_hubbard_holstein_model()

# --------------------------------------------------------------
# 
#    PSI4 ELECTRONIC CALC.
#
# --------------------------------------------------------------

def _run_elec_psi4_quantum_pipeline():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t USING PSI4 CALCULATION DATA")
        log.info("\t START MOLECULAR QUANTUM PIPELINE")
        log.info("\n")
    #
    # temporary placeholder:
    # keep current behavior here until we introduce
    # build_molecular_problem() and run_adapt()
    #
    calc_handler = run_psi4_electronic_structure()
    #
    # current qubit conversion / quantum pipeline hook
    #
    Qcalc_handler = QSIM(calc_handler)

#
#   CODE usage
#

def _print_usage():
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.warning("\t CODE USAGE: \n")
        log.warning(
            "\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] "
            "-ct2 [inhomo,homo,full | HH_MODEL,PSI4,PYSCF] - yml_inp [input]"
        )
        log.info("\t " + p.sep)
