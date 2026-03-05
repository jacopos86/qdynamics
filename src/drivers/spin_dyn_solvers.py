import os
from src.set_param_object import p
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.spin_model.spin_hamiltonian import set_spin_hamiltonian, quantum_spin_hamiltonian
from src.spin_model.nuclear_spin_config import nuclear_spins_config
#
def compute_spinsys_dephas(ZFS_CALC, HFI_CALC):
    #
    # main driver code for the calculation
    # of spin dephasing using a quantum algorithm
    # based on Suzuki-Trotter evolution
    #
    # first set atoms + index maps
    atoms.set_atoms_data()
    # check restart exists otherwise create one
    if not os.path.isdir(p.write_dir+'/restart'):
        if mpi.rank == mpi.root:
            os.mkdir(p.write_dir+'/restart')
    mpi.comm.Barrier()
    # extract interaction gradients
    interact_dict = calc_interaction_grad(ZFS_CALC, HFI_CALC)
    mpi.comm.Barrier()
    # n. atoms
    nat = atoms.nat
    # extract unperturbed struct.
    if mpi.rank == mpi.root:
        log.info("\t GS DATA DIR: " + p.gs_data_dir)
        log.info("\t fermion -> qubit transformation: " + str(p.fermion2qubit))
    struct_0 = build_gs_spin_struct(p.gs_data_dir, HFI_CALC)
    # set nuclear spin configuration
    nuclear_config = None
    if HFI_CALC:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t number nuclear spins: " + str(p.nsp))
            log.info("\t nuclear config. index: " + str(config_index))
            log.info("\t " + p.sep)
            log.info("\n")
        # set spin config.
        nuclear_config = nuclear_spins_config(p.nsp, p.B0)
        nuclear_config.set_nuclear_spins(nat, config_index)
    # qubitize the Hamiltonian
    Hsys = quantum_spin_hamiltonian(p.fermion2qubit, NUCL_SPINS=HFI_CALC, PHONONS=p.qubit_ph)
    Hsys.set_system_qubit_hamiltonian(struct_0, p.B0, nuclear_config)
    if mpi.rank == mpi.root:
        Hsys.print_info()
    Hsys.qubitize_spin_hamilt()