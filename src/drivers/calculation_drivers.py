from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.utilities.input_parser import parser
from src.set_param_object import p
from src.ground_state_solvers import solve_Hubbard_model, PSI4_elec_gs_driver
from src.spin_dyn_solvers import compute_spinsys_dephas

#
#  spin qubit driver
#

def spin_system_driver(yml_file):
    # prepare spin dephasing calculation
    calc_type1 = parser.parse_args().ct1[0]
    calc_type2 = parser.parse_args().ct2
    if calc_type1 == "GS":
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN SYSTEM CALCULATION")
            log.info("\n")
        # ------------------------------------------------------------
        #
        #    CHECK calc_type2 variable
        #
        # ------------------------------------------------------------
        if calc_type2 == "homo":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
                log.info("\n")
            # --------------------------------------------------------------
            # 
            #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = True
            HFI_CALC = False
            if mpi.rank == mpi.root:
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS SPIN - DEPHASING")
                log.info("\t ZFS_CALC: " + str(ZFS_CALC))
                log.info("\t HFI_CALC: " + str(HFI_CALC))
                log.info("\n")
                log.info("\t " + p.sep)
        elif calc_type2 == "inhomo":
            # --------------------------------------------------------------
            # 
            #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
            #
            # --------------------------------------------------------------
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
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] -ct2 [inhomo,homo,full | MODEL,PSI4,PYSCF] - yml_inp [input]")
                log.info("\t " + p.sep)
            log.error("\t calc_type2 wrong: " + calc_type2)
        #
        #    read input file
        #
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_calc_handler = compute_spinsys_ground_state(ZFS_CALC, HFI_CALC)
        #
        # finalize calculation
        #
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t" + p.sep)
            log.info("\t PRINT DATA ON FILES")
            # write T2 yaml files
            T2_calc_handler.print_results()
            log.info("\t" + p.sep)
            log.info("\n")
        mpi.comm.Barrier()
    # --------------------------------------------------------------
    # 
    #    FULL CALC. -> REAL TIME
    #
    # --------------------------------------------------------------
    elif calc_type1 == "RT":
        if mpi.rank == mpi.root:
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t REAL TIME DYNAMICS")
            log.info("\n")
            log.info("\t " + p.sep)
        # read input file
        p.read_yml_data(yml_file)
        p.check_consistency()
        # compute dephas
        if calc_type2 == "homo":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
                log.info("\n")
            # --------------------------------------------------------------
            # 
            #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = True
            HFI_CALC = False
        elif calc_type2 == "inhomo":
            # --------------------------------------------------------------
            # 
            #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = False
            HFI_CALC = True
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
                log.info("\n")
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
        #
        #  START NON MARKOVIAN CALCULATION
        #
        T2_calc_handler = compute_spinsys_dephas(ZFS_CALC, HFI_CALC)
    else:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CODE USAGE: \n")
            log.warning("\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] -ct2 [inhomo,homo,full | MODEL,PSI4,PYSCF] - yml_inp [input]")
            log.info("\t " + p.sep)
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")

#
#    electron system calculation driver
#

def elec_system_driver(yml_file):
    # prepare spin dephasing calculation
    calc_type1 = parser.parse_args().ct1[0]
    calc_type2 = parser.parse_args().ct2
    # ground state solver
    if calc_type1 == "GS":
        if mpi.rank == mpi.root:
            log.info("\t GROUND STATE CALCULATION -> STARTING")
            log.info("\n")
            log.info("\t " + p.sep)
        # read input file
        p.read_yml_data(yml_file)
        # Hubbard model
        if calc_type2 == "MODEL":
            #
            #  START MODEL CALCULATION
            #
            T2_calc_handler = solve_Hubbard_model()
        elif calc_type2 == "PSI4":
            # --------------------------------------------------------------
            # 
            #    RUN PSI4 driver
            #
            # --------------------------------------------------------------
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t USING PSI4 CALCULATION DATA")
                log.info("\n")
            #
            #  START NON MARKOVIAN CALCULATION
            #
            T2_calc_handler = PSI4_elec_gs_driver()
        else:
            if mpi.rank == mpi.root:
                log.warning("\t GROUND STATE SOLVER")
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")
    # RT solver
    if calc_type1 == "RT":
        if mpi.rank == mpi.root:
            log.info("\t ELECTRON DYNAMICS CALCULATION -> STARTING")
            log.info("\t REAL TIME DYNAMICS")
            log.info("\n")
            log.info("\t " + p.sep)
        # read input file
        p.read_yml_data(yml_file)
        #
        #  START DYNAMICAL CALCULATION
        #
        T2_calc_handler = elec_dyn_driver()
    else:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CODE USAGE: \n")
            log.warning("\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] -ct2 [inhomo,homo,full | MODEL,PSI4,PYSCF] - yml_inp [input]")
            log.info("\t " + p.sep)
        log.error("\t WRONG ACTION FLAG TYPE: QDYNAMICS STOPS HERE")