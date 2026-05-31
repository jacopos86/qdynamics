import os
import psi4
from pathlib import Path
from src.parameters.set_param_object import p
from src.utilities.log import log
from src.backends.psi4.molecular_structure import geometry_optimization
from src.backends.psi4.psi4_wfc_module import Psi4Wavefunction
from src.electrons.elec_hamiltonian import ElectronicHamiltonian
from src.electrons.density_matrix import DensityMatrix
from src.electrons.molecular_orbitals import molecular_orbitals_class
from src.basis_func.atomic_overlap import AO_overlap_class
from src.elec_boson.elec_vibr_coupl import ElecVibrCoupling
from src.common.units import Q_

#
#     PSI4 MAIN DRIVER CLASS
# 

class Psi4Driver:
    def __init__(self, basis_set_file, calc_parameters):
        basis_set_file = Path(basis_set_file)
        self.basis_set_file = Path(p.work_dir) / basis_set_file
        # PSI4 calc. parameters
        self.scf_type = calc_parameters.get("scf_type")
        self.scf_mode = calc_parameters.get("reference")
        self.method = calc_parameters.get("method")
        self.e_converg = calc_parameters.get("e_converg")
        self.d_converg = calc_parameters.get("d_converg")
        self.maxiter = calc_parameters.get("max_iter")
        self.guess = calc_parameters.get("guess")
        self.soscf = calc_parameters.get("soscf")
        self.soscf_max_iter = calc_parameters.get("soscf_max_iter")
    #
    #    set up the calculation parameters
    #
    def set_up_calc_parameters(self):
        basis_file = Path(self.basis_set_file).resolve()
        basis_dir = str(basis_file.parent)
        basis_name = basis_file.stem
        os.environ["PSIPATH"] = basis_dir + ":" + os.environ.get("PSIPATH", "")
        #   PSI4 calculation options
        psi4.set_options({
            'basis': basis_name,
            'scf_type': self.scf_type,
            'reference': self.scf_mode,
            'e_convergence': self.e_converg,
            'd_convergence': self.d_converg,
            'maxiter': self.maxiter,
            'guess': self.guess,
            'soscf': self.soscf,
            'soscf_max_iter': self.soscf_max_iter,
        })
        log.info(
            "\t frozen core: %s",
            psi4.core.get_global_option('FREEZE_CORE')
        )
    #
    #     SET UP
    #     ELECTRONIC STRUCTURE DATA 
    #
    def set_electronic_operators(self, WF):
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t SET ELECTRONIC STRUCTURE OBJECTS")
        log.info("\t " + p.sep)
        log.info("\n")
        # molecular integrals
        mints = WF.mints()
        # atomic overlaps
        S_obj = AO_overlap_class()
        S_obj.build_from_WF(WF)
        # molecular orbitals
        MO_obj = molecular_orbitals_class()
        MO_obj.set_mo_from_WF_obj(WF)
        # set density matrix
        DM_obj = DensityMatrix()
        DM_obj.set_occup_from_wfn(WF)
        DM_obj.set_orbital_occup(MO_obj)
        DM_obj.compute_dm_from_mo(MO_obj)
        # set electronic Hamiltonian object
        He = ElectronicHamiltonian()
        He.set_nuclear_repulsion_energy(WF.mol_struct.geometry)
        He.set_AO_operators(mints)
        return S_obj, MO_obj, DM_obj, He
    def build_operators_MO_basis(self, MO_obj, DM_obj, He):
        DM_obj.set_ae_space_dm(MO_obj.nmo)
        # set 1p matrix elements
        He.set_ae_1p_matr_elements(MO_obj)
    def set_elecvibr_inter(self, WF):
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t SET ELECTRON VIBRON COUPLING")
        log.info("\t " + p.sep)
        log.info("\n")
        # molecular integrals
        evibr = ElecVibrCoupling()
        evibr.set_AO_operators(WF)
    #
    #     PSI4 geometry driver
    #
    def psi4_geometry_driver(self, 
                coordinate_file, 
                optimized_coordinate_file, 
                charge, 
                multiplicity
        ):
        self.set_up_calc_parameters()
        # optimize geometry
        mol_struct = geometry_optimization(
            coordinate_file, 
            optimized_coordinate_file,
            charge, 
            multiplicity,
            self.method
        )
        return mol_struct
    #
    #     PSI4 electronic ground state
    #     driver
    #
    def compute_elec_struct(self, WF):
        """
        Run the actual Psi4 method and store the converged wavefunction.
        """
        energy, WF.wfn = psi4.energy(
            self.method,
            molecule=WF.mol_struct.geometry,
            return_wfn=True,
        )
        if WF.wfn.nirrep() != 1:
            raise ValueError(
                f"Wavefunction has nirrep={WF.wfn.nirrep()}, expected 1. "
                "Use symmetry c1 in the geometry."
            )
        WF.energy = Q_(energy, "hartree")
        WF.energy_1p = Q_(WF.wfn.variable("ONE-ELECTRON ENERGY"), "hartree")
        WF.energy_2p = Q_(WF.wfn.variable("TWO-ELECTRON ENERGY"), "hartree")
    #
    #    TESTS
    #
    def run_consistency_tests(self, WF, S_obj, MO_obj, DM_obj):
        S_obj.run_tests()
        MO_obj.run_tests(S_obj.S)
        DM_obj.run_tests(S_obj.S, WF.mol_struct.nel)
    def energy_report(self, He, WF, DM_obj):
        He.check_total_energy(WF, DM_obj)
    #
    #    PSI4 main driver routine
    #
    def psi4_elec_struct_driver(self, mol_struct):
        WF = Psi4Wavefunction(mol_struct)
        self.compute_elec_struct(WF)
        log.info("\n")
        WF.print_summary()
        log.info("\t " + p.sep)
        return WF