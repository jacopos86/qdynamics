import os
from pathlib import Path

import basis_set_exchange
import numpy as np
import psi4

from src.common.units import Q_
from src.io_module.write_xyz_file import write_xyz
from src.utilities.log import log
from src.chemistry.psi4.electron_phonon_solver import ElectronPhononSolver


special_basis = {"SBKJC": "SBKJC-VDZ"}
SEP = "*" * 94


def _work_path(path, work_dir=None):
    if work_dir is None:
        work_dir = Path.cwd()
    return Path(work_dir) / Path(path)


def write_basis_lib_file(basis_file, unique_elements, basis_map=None, work_dir=None):
    """
    Write a Psi4-compatible BASIS library file for the given elements.
    """
    basis_file = Path(basis_file)
    basis_file = _work_path(basis_file, work_dir)
    # delete existing file if present
    if basis_file.exists():
        return
    if basis_map is None:
        raise ValueError("basis_map is required when writing a basis-set file")
    if isinstance(basis_map, str):
        basis_map = {symbol: basis_map for symbol in unique_elements}
    log.info("\t BASIS SET:")
    # iterate over elements
    for symbol in sorted(unique_elements):
        basis_name = basis_map.get(symbol)
        log.info(f"\t\t {symbol}: {basis_name}")
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
        log.info(f"\t\t " + basis_str)
        # append to file
        with basis_file.open("a") as f:
            f.write(basis_str)
            f.write("\n")
    log.info("\t " + SEP)
    log.info("\n")


def setup_basis_set(coord_file, basis_file_name, basis_map=None, work_dir=None):
    """
    Read a XYZ coordinate file and write a basis-set library
    for the unique atomic species found.
    """
    coord_file = Path(coord_file)
    coord_file = _work_path(coord_file, work_dir)
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
    write_basis_lib_file(basis_file_name, sorted(elements), basis_map, work_dir)


class Psi4Molecule:
    """
    PSI4 molecular class
    """
    def __init__(self, xyz_file, charge, multiplicity, work_dir=None):
        self.xyz_file = Path(xyz_file)
        self.xyz_file = _work_path(self.xyz_file, work_dir)
        self.charge = charge
        self.multiplicity = multiplicity
        # geometry
        self.geometry = self._load_geometry()
        self.nel = self._compute_electrons()
        self.print_info_data()

    def _load_geometry(self):
        """
        load geometry in PSI4:
        remove symmetry line -> no symmetry used for now
        """
        with open(self.xyz_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 3:
            log.error(f"XYZ file too short: {self.xyz_file}")
        # first line: number of atoms
        try:
            nat = int(lines[0])
        except ValueError:
            log.error(f"First line must be number of atoms, got: {lines[0]}")
        # second line: units/comment
        unit_line = lines[1].lower()
        if unit_line in ["angstrom", "ang", "a"]:
            psi4_units = "angstrom"
        elif unit_line in ["bohr", "au", "a.u."]:
            psi4_units = "bohr"
        else:
            log.error(
                f"Unsupported unit line '{lines[1]}' in {self.xyz_file}. "
                "Use 'Angstrom' or 'Bohr'."
            )
        # remaining lines: atoms
        atom_lines = lines[2:]
        if len(atom_lines) != nat:
            log.error(
                f"XYZ atom count mismatch: header says {nat}, found {len(atom_lines)} atom lines"
            )
        geom = f"{self.charge} {self.multiplicity}\n"
        geom += "\n".join(atom_lines) + "\n"
        geom += f"units {psi4_units}\n"
        geom += f"symmetry c1\n"
        log.info("\t " + SEP)
        log.info("\t GEOM SENT TO PSI4:")
        log.info("\n"+geom)
        return psi4.geometry(geom)

    def _compute_electrons(self):
        Z = sum(self.geometry.Z(i) for i in range(self.geometry.natom()))
        return int(Z - self.charge)

    def compute_nuclear_repulsion_energy(self):
        return Q_(self.geometry.nuclear_repulsion_energy(), "hartree")

    def print_info_data(self):
        log.info(f"\t Total number atoms: {self.geometry.natom()}")
        log.info(f"\t Total num. electrons: {self.nel}")
        log.info(f"\t Multiplicity: {self.geometry.multiplicity()}")
        log.info(f"\t System charge: {self.geometry.molecular_charge()}")
        log.info(f"\t Repulsion energy: {self.compute_nuclear_repulsion_energy()}")


def geometry_optimization(
        coordinate_file,
        optimized_coordinate_file,
        charge,
        multiplicity,
        method,
        work_dir=None
):
    """
    geometry optimization interface
    """
    atom_struct = Psi4Molecule(coordinate_file, charge, multiplicity, work_dir)
    # set optimization run
    E_SCF, wfn = psi4.optimize(
        method.lower(),
        molecule=atom_struct.geometry,
        return_wfn=True
    )
    # save data on file
    if work_dir is None:
        work_dir = Path.cwd()
    write_xyz(atom_struct, optimized_coordinate_file, work_dir)
    atom_struct.print_info_data()
    return atom_struct


def geometry_from_input(
        coordinate_file,
        charge,
        multiplicity,
        work_dir=None
):
    """
    Build a Psi4 molecule from the input coordinates without optimization.
    """
    atom_struct = Psi4Molecule(coordinate_file, charge, multiplicity, work_dir)
    atom_struct.print_info_data()
    return atom_struct


class Psi4Wavefunction:
    """
    Wrapper around psi4.core.Wavefunction.
    """
    def __init__(self, mol_struct: Psi4Molecule):
        self.mol_struct = mol_struct
        # energy / wfn
        self.wfn = None
        self.energy = None
        self.energy_1p = None
        self.energy_2p = None
        self._build_empty()

    def _build_empty(self):
        """
        Build an empty wavefunction container.
        Useful for MintsHelper / AO integrals before running SCF.
        """
        basis_name = psi4.core.get_global_option("BASIS")
        self.wfn = psi4.core.Wavefunction.build(self.mol_struct.geometry, basis_name)
        # check wfn
        if self.wfn.nirrep() != 1:
            raise ValueError(
                f"Wavefunction has nirrep={self.wfn.nirrep()}, expected 1. "
                "Use symmetry c1 in the geometry."
            )

    def mints(self):
        """
        Return MintsHelper from the current wavefunction.
        """
        if self.wfn is None:
            log.error("Wavefunction not available. Call build_empty() or compute().")
        return psi4.core.MintsHelper(self.wfn.basisset())

    def nalpha(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nalpha()

    def nbeta(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nbeta()

    def nirrep(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nirrep()

    def nmo(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.nmo()

    def Ca(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.Ca()

    def Cb(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        return self.wfn.Cb()

    def print_summary(self):
        if self.wfn is None:
            log.error("Wavefunction not available.")
        log.info("\t Psi4 wavefunction summary")
        log.info(f"\t nirrep      : {self.wfn.nirrep()}")
        log.info(f"\t nalpha      : {self.wfn.nalpha()}")
        log.info(f"\t nbeta       : {self.wfn.nbeta()}")
        log.info(f"\t nmo         : {self.wfn.nmo()}")
        if self.energy is not None:
            log.info(f"\t energy      : {self.energy}")
        if self.energy_1p is not None:
            log.info(f"\t energy one-particle      : {self.energy_1p}")
        if self.energy_2p is not None:
            log.info(f"\t energy two-particle      : {self.energy_2p}")


class AO_overlap_class:
    """
    AO overlap matrix and its first derivatives
    """
    def __init__(self, toler=1.0e-6):
        # overlap matrix
        self.S = None
        self.gradS = None
        self.TOLER = toler
        # num. basis func.
        self.nbf = None
        self.natom = None

    def build_from_WF(self, WF, with_gradient=True):
        """
        Build overlap matrix and optionally its nuclear derivatives
        from a Wavefunction class.
        """
        mints = WF.mints()
        self.natom = WF.mol_struct.geometry.natom()
        self.nbf = WF.wfn.basisset().nbf()
        # set overlap matrix
        self.set_overlap_matr(mints)
        # set atomic gradients
        if with_gradient:
            self.set_overlap_matr_grad(mints)
        # info
        self.print_info()

    def set_overlap_matr(self, mints):
        """
        Read AO overlap matrix from Psi4 MintsHelper.
        """
        S = np.asarray(mints.ao_overlap())
        if S.shape != (S.shape[0], S.shape[0]):
            log.error(f"Overlap matrix is not square: shape={S.shape}")
        assert self.nbf == S.shape[0]
        self.S = S.copy()

    def set_overlap_matr_grad(self, mints):
        """
        Read first derivatives of AO overlap matrix:
            gradS[xyz, atom, mu, nu]
        """
        if self.nbf is None or self.natom is None:
            log.error("nbf/natoms not initialized. Call build_from_mints first.")
        self.gradS = np.zeros((3, self.natom, self.nbf, self.nbf))
        for ia in range(self.natom):
            gS = mints.ao_oei_deriv1("OVERLAP", ia)
            for idx in range(3):
                A = np.asarray(gS[idx])
                if A.shape != (self.nbf, self.nbf):
                    log.error(
                        f"Gradient overlap shape mismatch for atom {ia}, dir {idx}: "
                        f"{A.shape} != ({self.nbf}, {self.nbf})"
                    )
                self.gradS[idx, ia, :, :] = A

    def _check_overlap_matr_properties(self):
        """
        Check:
         - diagonal close to 1
         - symmetry
         - no element larger than 1 by more than tolerance
        """
        if self.S is None:
            log.error("Overlap matrix S is not initialized")
        # diagonal
        diag_err = np.max(np.abs(np.diag(self.S) - 1.0))
        # symmetry
        sym_err = np.max(np.abs(self.S - self.S.T))
        # optional sanity check
        max_elem = np.max(np.abs(self.S))
        if diag_err > self.TOLER:
            log.error(f"Overlap diagonal deviates from 1 (max error = {diag_err:.3e})")
        if sym_err > self.TOLER:
            log.error(f"Overlap matrix is not symmetric (max error = {sym_err:.3e})")
        if max_elem > 1.0 + self.TOLER:
            log.error(f"Overlap matrix contains element > 1 (max abs element = {max_elem:.6f})")
        log.info(
            f"\t AO overlap matrix checks passed "
            f"(diag err = {diag_err:.3e}, sym err = {sym_err:.3e})"
        )

    def print_info(self):
        if self.S is None:
            log.warning("AO overlap matrix not initialized")
            return
        log.info(f"\t nbf: {self.nbf}")
        log.info(f"\t S shape: {self.S.shape}")
        if self.gradS is not None:
            log.info(f"\t gradS shape: {self.gradS.shape}")
        log.info("\t " + SEP)

    def run_tests(self):
        self._check_overlap_matr_properties()


class MolecularOrbitals:
    """
    Molecular orbitals class
    """
    def __init__(self):
        self.C = None
        # up / down
        self.Ca = None
        self.Cb = None
        # n. MOs
        self.nmo = None

    def set_mo_from_WF_obj(self, WF):
        """
        Set molecular orbitals from a Psi4 Wavefunction object.
        Stores AO->MO coefficient matrices as NumPy arrays.
        """
        if WF is None:
            log.error("WF object is None")
        # n. MOs
        self.nmo = WF.nmo()
        self.Ca = np.asarray(WF.Ca())
        self.Cb = np.asarray(WF.Cb())
        if self.Ca.shape[1] != self.nmo:
            log.error(
                f"Inconsistent alpha MO shape: {self.Ca.shape}, expected nmo={self.nmo}"
            )
        if self.Cb.shape[1] != self.nmo:
            log.error(
                f"Inconsistent beta MO shape: {self.Cb.shape}, expected nmo={self.nmo}"
            )
        # closed-shell case: Ca == Cb
        if np.allclose(self.Ca, self.Cb):
            self.C = self.Ca.copy()
        else:
            self.C = None

    def _check_orthogonality(self, S):
        """
        Check MO orthogonality:
            C^T S C = I
        """
        S = np.asarray(S)
        if self.C is not None:
            self._check_matrix(self.C, S, label="Restricted MOs")
        else:
            if self.Ca is None or self.Cb is None:
                log.error("MO coefficients not initialized")
            self._check_matrix(self.Ca, S, label="Alpha MOs")
            self._check_matrix(self.Cb, S, label="Beta MOs")

    def _check_matrix(self, C, S, label, tol=1.e-8):
        M = C.T @ S @ C
        I = np.eye(M.shape[0])
        err = np.max(np.abs(M - I))
        if err > tol:
            log.error(f"{label}: C^T S C != I  (max error = {err:.3e})")
        else:
            log.info(f"\t {label}: C^T S C = I  (max error = {err:.3e})")

    def run_tests(self, S):
        self._check_orthogonality(S)


molecular_orbitals_class = MolecularOrbitals


class DensityMatrix:
    """
    Density matrix container.
    """
    def __init__(self):
        # AO density matrices
        self.Dao = None          # [Da, Db]
        # MO-space density matrices
        self.Dact = None         # [Dact_a, Dact_b]
        self.Dae = None          # [Dae_a, Dae_b]
        # occupations
        self.nocc = np.zeros(2, dtype=int)
        self.nel = None
        self.f = None            # restricted spin-orbital occupations
        self.fa = None           # alpha spin-orbital occupations
        self.fb = None           # beta  spin-orbital occupations

    def set_occup_from_wfn(self, WF):
        """
        Set occupied orbital counts from Psi4 wavefunction.
        """
        self.nocc[0] = WF.nalpha()
        self.nocc[1] = WF.nbeta()
        self.nel = sum(self.nocc)

    def set_orbital_occup(self, MO_obj):
        """
        Build spin-orbital occupation arrays.
        """
        nmo = MO_obj.nmo
        if MO_obj.C is not None:
            self.f = np.zeros(2 * nmo)
            for i in range(self.nocc[0]):
                self.f[2 * i] = 1.0
                self.f[2 * i + 1] = 1.0
        elif MO_obj.Ca is not None or MO_obj.Cb is not None:
            self.fa = np.zeros(2 * nmo)
            self.fb = np.zeros(2 * nmo)
            for i in range(self.nocc[0]):
                self.fa[2 * i] = 1.0
            for i in range(self.nocc[1]):
                self.fb[2 * i + 1] = 1.0
        else:
            log.error("MO_obj is not initialized")

    def compute_dm_from_mo(self, MO_obj):
        """
        Build AO density matrices from MO coefficients.
        Dao = [Da, Db]
        """
        # restricted
        if MO_obj.C is not None:
            C = np.asarray(MO_obj.C)
            Da = np.einsum("mi,ni->mn", C[:, :self.nocc[0]], C[:, :self.nocc[0]])
            self.Dao = [Da, Da.copy()]
        # unrestricted / ROHF
        else:
            if MO_obj.Ca is None or MO_obj.Cb is None:
                log.error("MO coefficients not initialized")
            Ca = np.asarray(MO_obj.Ca)
            Cb = np.asarray(MO_obj.Cb)
            Da = np.einsum("mi,ni->mn", Ca[:, :self.nocc[0]], Ca[:, :self.nocc[0]])
            Db = np.einsum("mi,ni->mn", Cb[:, :self.nocc[1]], Cb[:, :self.nocc[1]])
            self.Dao = [Da, Db]

    def set_ae_space_dm(self, nmo):
        """
        Build full-space diagonal density matrix in spin-orbital MO basis.
        """
        self.Dae = np.zeros((2*nmo, 2*nmo))
        if self.f is not None:
            for i in range(self.f.shape[0]):
                self.Dae[i, i] = self.f[i]
        else:
            for i in range(self.fa.shape[0]):
                self.Dae[i, i] = self.fa[i]
                self.Dae[i, i] += self.fb[i]

    def _compare_total_electron_number(self, S, nel_expected, tol=1.0e-6):
        """
        Check Tr(S D) = number of electrons.
        """
        if self.Dao is None:
            log.error("AO density matrix not initialized")
        S = np.asarray(S)
        nel_up = np.trace(S @ self.Dao[0])
        nel_dw = np.trace(S @ self.Dao[1])
        nel = nel_up + nel_dw
        log.info(f"\t Tr(S Da) = {nel_up}")
        log.info(f"\t Tr(S Db) = {nel_dw}")
        err = abs(nel - nel_expected)
        log.info(f"\t Total electrons from density = {nel}")
        log.info(f"\t Expected electrons           = {nel_expected}")
        if err > tol:
            log.error(f"Electron number check failed (error = {err:.3e})")
        psi4.compare_values(nel, nel_expected, 4, 'number of electrons')

    def run_tests(self, S, nel_expected):
        self._compare_total_electron_number(S, nel_expected)


class ElectronicHamiltonian:
    """
    Electronic Hamiltonian class
    """
    def __init__(self):
        # ACTIVE space objects
        self.H1p = None
        self.H2p = None
        # Vnn
        self.Vnn = None
        # TEI -> MO basis
        self.Iijkl = None
        # OEI -> MO basis
        self.hij = None
        # AO basis
        self.T_ao = None
        self.V_ao = None
        self.H0_ao = None
        self.Hee_ao = None

    # set nuclear interaction energy
    def set_nuclear_repulsion_energy(self, geometry):
        self.Vnn = Q_(
            geometry.nuclear_repulsion_energy(),
            "hartree"
        )

    # AO basis set
    def set_AO_operators(self, mints):
        self.set_ao_kinetic_operator(mints)
        self.set_ao_one_part_potential(mints)
        self.set_ao_one_part_hamiltonian()
        self.set_ao_two_part_hamiltonian(mints)

    # set AO kinetic operator
    def set_ao_kinetic_operator(self, mints):
        self.T_ao = np.asarray(mints.ao_kinetic())

    # set AO one particle potential
    def set_ao_one_part_potential(self, mints):
        self.V_ao = np.asarray(mints.ao_potential())

    # set AO one particle hamiltonian
    def set_ao_one_part_hamiltonian(self):
        self.H0_ao = self.T_ao + self.V_ao

    # set AO two particle hamiltonian
    def set_ao_two_part_hamiltonian(self, mints):
        self.Hee_ao = np.asarray(mints.ao_eri())

    def set_ae_1p_matr_elements(self, MO_obj):
        """
        Transform one-particle AO Hamiltonian to MO basis.
        """
        if self.H0_ao is None:
            log.error("AO one-particle Hamiltonian H0_ao is not initialized")
        _hij = [None, None]
        # Restricted case
        if MO_obj.C is not None:
            C = np.asarray(MO_obj.C)
            _hij[0] = C.T @ self.H0_ao @ C
            _hij[1] = _hij[0].copy()
        # Unrestricted / ROHF case
        else:
            if MO_obj.Ca is None or MO_obj.Cb is None:
                log.error("MO coefficients are not initialized")
            Ca = np.asarray(MO_obj.Ca)
            Cb = np.asarray(MO_obj.Cb)
            _hij[0] = Ca.T @ self.H0_ao @ Ca
            _hij[1] = Cb.T @ self.H0_ao @ Cb
        # set full spin orbital matrix
        self.hij = np.zeros((2*MO_obj.nmo, 2*MO_obj.nmo))
        for i in range(MO_obj.nmo):
            for j in range(MO_obj.nmo):
                self.hij[2*i, 2*j] = _hij[0][i, j]
                self.hij[2*i+1, 2*j+1] = _hij[1][i, j]

    def set_ae_2p_matr_elements(self, MO_obj):
        """
        Transform AO electron-repulsion integrals to spin-orbital MO basis.
        """
        if self.Hee_ao is None:
            log.error("AO two-particle Hamiltonian Hee_ao is not initialized")
        # internal num. molecular orbitals
        nmo = MO_obj.nmo
        Iao = np.asarray(self.Hee_ao)
        coeffs = [np.asarray(MO_obj.Ca), np.asarray(MO_obj.Cb)]
        # I_ijkl molecular orbital basis
        self.Iijkl = np.zeros((2*nmo, 2*nmo, 2*nmo, 2*nmo))
        # loop over spin blocks
        for spin_ij in range(2):
            Cij = coeffs[spin_ij]
            for spin_kl in range(2):
                Ckl = coeffs[spin_kl]
                block = np.einsum(
                    "mi,nj,pk,ql,mnpq->ijkl",
                    Cij,
                    Cij,
                    Ckl,
                    Ckl,
                    Iao,
                    optimize=True,
                )
                i_idx = slice(spin_ij, 2*nmo, 2)
                j_idx = slice(spin_ij, 2*nmo, 2)
                k_idx = slice(spin_kl, 2*nmo, 2)
                l_idx = slice(spin_kl, 2*nmo, 2)
                self.Iijkl[i_idx, j_idx, k_idx, l_idx] = block
        self.H2p = self.Iijkl

    def check_total_energy(self, WF, DM_obj):
        _tot_energy = WF.energy_1p.magnitude + WF.energy_2p.magnitude + self.Vnn.magnitude
        psi4.compare_values(WF.energy.magnitude, _tot_energy, 6, 'total energy')
        self._check_1p_energy(WF, DM_obj)
        self._check_2p_energy(WF, DM_obj)

    def _check_1p_energy(self, WF, DM_obj):
        D = DM_obj.Dae
        _en_1p = np.trace(D @ self.hij)
        en_1p = Q_(_en_1p, "hartree")
        psi4.compare_values(en_1p.magnitude, WF.energy_1p.magnitude, 6, 'one particle energy')

    def _check_2p_energy(self, WF, DM_obj):
        occ = np.diag(DM_obj.Dae)
        en_2p = 0.0
        for p in range(occ.shape[0]):
            if occ[p] == 0.0:
                continue
            for q in range(occ.shape[0]):
                if occ[q] == 0.0:
                    continue
                en_2p += 0.5 * occ[p] * occ[q] * (
                    self.Iijkl[p, p, q, q] - self.Iijkl[p, q, q, p]
                )
        psi4.compare_values(en_2p, WF.energy_2p.magnitude, 6, 'two particle energy')


class Psi4Driver:
    def __init__(self, basis_set_file, calc_parameters, work_dir=None):
        basis_set_file = Path(basis_set_file)
        self.work_dir = work_dir
        self.basis_set_file = _work_path(basis_set_file, work_dir)
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

    def set_up_calc_parameters(self):
        basis_file = Path(self.basis_set_file).resolve()
        basis_dir = str(basis_file.parent)
        basis_name = basis_file.stem
        os.environ["PSIPATH"] = basis_dir + ":" + os.environ.get("PSIPATH", "")
        # PSI4 calculation options
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

    def set_electronic_operators(self, WF):
        log.info("\n")
        log.info("\t " + SEP)
        log.info("\t SET ELECTRONIC STRUCTURE OBJECTS")
        log.info("\t " + SEP)
        log.info("\n")
        # molecular integrals
        mints = WF.mints()
        # atomic overlaps
        S_obj = AO_overlap_class()
        S_obj.build_from_WF(WF)
        # molecular orbitals
        MO_obj = MolecularOrbitals()
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
        # set 2p matrix elements
        He.set_ae_2p_matr_elements(MO_obj)

    def set_elecvibr_inter(self, WF):
        log.info("\n")
        log.info("\t " + SEP)
        log.info("\t SET ELECTRON VIBRON COUPLING")
        log.info("\t " + SEP)
        log.info("\n")
        # molecular integrals
        evibr = ElectronPhononSolver()
        evibr.set_AO_operators(WF)

    def psi4_geometry_driver(self,
                coordinate_file,
                optimized_coordinate_file,
                charge,
                multiplicity,
                optimize_geometry=True
        ):
        self.set_up_calc_parameters()
        if not optimize_geometry:
            log.info("\t Skipping geometry optimization; using input geometry.")
            return geometry_from_input(
                coordinate_file,
                charge,
                multiplicity,
                self.work_dir
            )
        # optimize geometry
        mol_struct = geometry_optimization(
            coordinate_file,
            optimized_coordinate_file,
            charge,
            multiplicity,
            self.method,
            self.work_dir
        )
        return mol_struct

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

    def run_consistency_tests(self, WF, S_obj, MO_obj, DM_obj):
        S_obj.run_tests()
        MO_obj.run_tests(S_obj.S)
        DM_obj.run_tests(S_obj.S, WF.mol_struct.nel)

    def energy_report(self, He, WF, DM_obj):
        He.check_total_energy(WF, DM_obj)

    def psi4_elec_struct_driver(self, mol_struct):
        WF = Psi4Wavefunction(mol_struct)
        self.compute_elec_struct(WF)
        log.info("\n")
        WF.print_summary()
        log.info("\t " + SEP)
        return WF
