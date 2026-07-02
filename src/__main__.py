from src.set_param_object import p
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.utilities.timer import timer
from src.utilities.input_parser import parser
from src.calculation_drivers import spin_system_driver, elec_system_driver
from pathlib import Path

def run():
    #
    # set up parallelization
    #
    if mpi.rank == mpi.root:
        log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        log.info("\t ++++++                                                                                  ++++++")
        log.info("\t ++++++                           QDYNAMICS   CODE                                       ++++++")
        log.info("\t ++++++                                                                                  ++++++")
        log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #
    #  input structure :
    #  log.warning("\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] -ct2 [inhomo,homo,full | MODEL,PSI4,PYSCF] - yml_inp [input]")
    # 
    yml_file_in = parser.parse_args().yml_inp[0]
    if yml_file_in is None:
        log.error("-> yml file name missing")
    else:
        yml_file = Path(yml_file_in).resolve()
        if not Path.exists(yml_file):
            log.error(f"-> yml file not found in: {yml_file}")
    nargs = 2
    if parser.parse_args().co is not None:
        nargs += 1
    if parser.parse_args().ct2 is not None:
        nargs += 1
    if nargs < 4:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CODE USAGE: \n")
            log.warning("\t -> python src -ct1 [GS, RT] -co [spin-sys | elec-sys] -ct2 [inhomo,homo,full | MODEL,PSI4,PYSCF] - yml_inp [input]")
        log.error("\t WRONG EXECUTION PARAMETERS: QDYNAMICS STOPS")
    else:
        if mpi.rank == mpi.root:
            log.debug("\t observable: " + str(parser.parse_args().co[0]))
            log.debug("\t calculation type (1): " + str(parser.parse_args().ct1[0]))
            log.debug("\t calculation type (2): " + str(parser.parse_args().ct2))
    timer.start_execution()

    #
    #  call different drivers
    #

    calc_type1 = parser.parse_args().ct1[0]
    if calc_type1 == "GS" or calc_type1 == "RT":
        co = parser.parse_args().co[0]
        if co == "spin-sys":
            spin_system_driver(yml_file)
        elif co == "elec-sys":
            elec_system_driver(yml_file)
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CALC. TYPE 1 NOT RECOGNIZED")
                log.warning("\t QUIT PROGRAM")
                log.info("\t " + p.sep)
            log.error("\t WRONG CALC. FLAG")
    else:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CALC. TYPE NOT RECOGNIZED")
            log.warning("\t QUIT PROGRAM")
            log.info("\t " + p.sep)
        log.error("\t WRONG CALC. FLAG")
    # end execution
    timer.end_execution()
    if mpi.rank == mpi.root:
        log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        log.info("\t ++++++                                                                                  ++++++")
        log.info("\t ++++++                    CALCULATION SUCCESSFULLY COMPLETED                            ++++++")
        log.info("\t ++++++                                                                                  ++++++")
        log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    PETSc.garbage_cleanup()

if __name__ == "__main__":
    run()
