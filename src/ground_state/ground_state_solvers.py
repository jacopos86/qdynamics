# This is the main subroutine
# for non Markovian calculation of electronic systems
#
import numpy as np
from src.parameters.set_param_object import p
from src.parallelization.mpi import mpi
from src.utilities.log import log
from src.backends.psi4.BasisSet_module import setup_basis_set
from src.backends.psi4.run import Psi4Driver
from src.hamiltonians.problem_hamiltonian import build_problem_hamiltonian

# ====================================================
#
#     Hubbard model solver
#
# ====================================================

def solve_Holstein_Hubbard_model():
    """
    Main static HH entrypoint.

    Repo-consistent role:
      1. resolve HH model inputs
      2. build HH Hamiltonian
      3. prepare initial state
      4. run GS / ADAPT-style static solve
      5. save JSON artifact
      6. optionally write PDF summary

    This is the static stage that later feeds realtime dynamics.
    """
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t COMPUTE GROUND-STATE HOLSTEIN-HUBBARD MODEL")
        log.info("\n")
        log.info("\t " + p.sep)
    # core HH parameters expected in repo workflow:
    # L, t, u, dv, omega0, g_ep, n_ph_max, boundary, ordering, boson_encoding
    L              = p.L
    t_hop          = p.t
    U              = p.U
    dv             = p.dv
    omega0         = p.w0
    g_ep           = p.g
    n_ph_max       = p.n_ph_max
    boundary       = p.boundary
    ordering       = p.ordering
    boson_encoding = p.boson_encoding
    fermion_encoding = p.fermion_encoding
    # static-solver / ADAPT settings
    adapt_max_depth = p.adapt_max_depth
    adapt_eps_grad  = p.adapt_eps_grad
    adapt_maxiter   = p.adapt_maxiter
    inner_optimizer = p.adapt_inner_optimizer
    initial_source  = p.initial_state_source
    # output paths
    output_json = p.output_json
    output_pdf  = getattr(p, "output_pdf", None)
    skip_pdf    = getattr(p, "skip_pdf", False)
    if mpi.rank == mpi.root:
        log.info(f"\t HH parameters: L={L}, t={t_hop}, U={U}, dv={dv}, omega0={omega0}, g_ep={g_ep}, n_ph_max={n_ph_max}")
        log.info(f"\t Boundary: {boundary}, Ordering: {ordering}, Boson encoding: {boson_encoding}, Fermion encoding: {fermion_encoding}")
        log.info(f"\t ADAPT settings: max_depth={adapt_max_depth}, eps_grad={adapt_eps_grad}, maxiter={adapt_maxiter}, inner_optimizer={inner_optimizer}")
        log.info(f"\t Initial state source: {initial_source}")
        log.info(f"\t Output JSON: {output_json}, Output PDF: {output_pdf} (skip_pdf={skip_pdf})")
    # --------------------------------------------
    # 2. build HH problem container
    # --------------------------------------------
    # repo guide:
    #   build_problem_hamiltonian(...)
    #   -> build_hubbard_holstein_hamiltonian(...)
    #   -> build_hh_sector_hamiltonian_ed(...)
    #
    # this stage resolves the static HH physics point used later
    hh_problem = build_problem_hamiltonian(
        problem_type="hh",
        L=L,
        t=t_hop,
        u=U,
        dv=dv,
        omega0=omega0,
        g_ep=g_ep,
        n_ph_max=n_ph_max,
        boundary=boundary,
        ordering=ordering,
        fermion_encoding=fermion_encoding,
        boson_encoding=boson_encoding,
    )
    # optional unpacking if your implementation keeps these separately
    H          = hh_problem.H
    basis      = hh_problem.basis
    metadata   = hh_problem.metadata
    observables = getattr(hh_problem, "observables", None)

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