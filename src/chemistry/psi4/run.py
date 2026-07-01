import psi4

#
#     PSI4 MAIN DRIVER CLASS
# 

class Psi4Driver:
    def __init__(self):
        self.basis_set_file = p.basis_set_file
        self.scf_type = p.scf_type
        self.scf_mode = p.scf_mode
        self.e_converg = p.e_converg
        self.d_converg = p.d_converg
        self.maxiter = p.max_iter
        self.guess = p.orbital_init
    #
    #    set up the calculation parameters
    #
    def set_up_calc_parameters(self):
        basis_name = self.basis_set_file[:-4]
        #   PSI4 calculation options
        psi4.set_options({
            'basis': basis_name,
            'scf_type': self.scf_type,
            'reference': self.scf_mode,
            'e_convergence': self.e_converg,
            'd_convergence': self.d_converg,
            'maxiter': self.maxiter,
            'guess': self.guess,
            'soscf': True,
            'soscf_max_iter': 40,
        })
        log.info(
            "frozen core: %s",
            psi4.core.get_global_option('FREEZE_CORE')
        )
    #
    #     PSI4 geometry driver
    #
    def psi4_geometry_driver(self):
        self.set_up_calc_parameters()
        # optimize geometry
        E_SCF, wfn = geometry_optimization()
        return E_SCF, wfn
    #
    #    PSI4 main driver routine
    #
    def psi4_main_driver(self, E_scf, wfn):
        # model initialization
        initialize_operators(wfn)
        # run tests
        run_consistency_tests()
        # finalize calculation -> perform MO basis conversion
        build_mo_basis_operators()
        # run energy tests
        energy_report(E_scf)