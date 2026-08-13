"""Finite-difference Psi4 mean-field electron--phonon coupling."""
import logging
import numpy as np
import psi4

log = logging.getLogger(__name__)
HARTREE_TO_WAVENUMBER = 219474.6313705
AMU_TO_ELECTRON_MASS = 1822.888486209


class FiniteDifferenceElectronPhononSolver:
    """Build Pulay-correct finite-difference EPH matrix elements.

    Psi4 1.9 does not expose all response terms needed to reproduce its SCF
    EPH derivative through the current Python API.  Until a native analytic
    implementation is available, the default ``pyscf`` engine evaluates the
    same RHF problem at the Psi4 geometry with :mod:`pyscf.eph.eph_fd`, whose
    finite-difference implementation includes the AO Pulay terms.  The old
    Psi4-only approximation remains available as ``engine="psi4_approx"`` for
    diagnostics, but it is not suitable for cross-code numerical comparison.
    """
    def __init__(self, wavefunction, method, fd_step=0.001,
                 cutoff_frequency=80.0, basis=None, engine="pyscf"):
        self.wfn = wavefunction
        self.mol = wavefunction.mol_struct.geometry
        self.method = method
        self.fd_step = float(fd_step)
        self.cutoff_frequency = float(cutoff_frequency)
        self.basis = basis
        self.engine = str(engine).lower()
        self.reference_basis = wavefunction.wfn.basisset()

    @staticmethod
    def _array(matrix):
        value = matrix.to_array() if hasattr(matrix, "to_array") else matrix
        # Psi4 1.9.x may expose a spin-block container rather than a single
        # ndarray for Fa().  RHF uses the first square block.
        if isinstance(value, (tuple, list)):
            blocks = [np.asarray(item) for item in value]
            square = [item for item in blocks if item.ndim == 2 and item.shape[0] == item.shape[1]]
            if square:
                # Psi4 returns one block per irreducible representation.
                # Reassemble the full AO/MO matrix in the corresponding
                # block order, including zero-dimensional blocks.
                n = sum(item.shape[0] for item in square)
                full = np.zeros((n, n), dtype=np.result_type(*square))
                offset = 0
                for block in square:
                    size = block.shape[0]
                    full[offset:offset + size, offset:offset + size] = block
                    offset += size
                value = full
        return np.asarray(value)

    def _displaced_fock(self, coords, symbols):
        mol = psi4.geometry("\n".join(
            ["0 1"] + [f"{s} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}"
                        for s, r in zip(symbols, coords)]
            + ["units bohr", "symmetry c1", "no_reorient", "no_com"]
        ))
        _, wfn = psi4.energy(self.method, molecule=mol, return_wfn=True)
        # Match the electronic operator differentiated by PySCF: F - T.
        fock = self._array(wfn.Fa())
        kinetic = self._array(wfn.mintshelper().ao_kinetic())
        return fock - kinetic

    def _run_pyscf(self):
        """Evaluate Pulay-correct RHF EPH at the exact Psi4 geometry."""
        if self.basis is None:
            raise ValueError(
                "A PySCF basis name/map is required for Pulay-correct EPH")
        if str(self.method).lower() not in ("scf", "hf", "rhf"):
            raise NotImplementedError(
                "The Pulay-correct compatibility engine currently supports "
                "closed-shell RHF/SCF only")
        if self.mol.multiplicity() != 1:
            raise NotImplementedError(
                "The Pulay-correct compatibility engine currently supports "
                "singlet closed-shell molecules only")
        from pyscf import gto, scf
        from pyscf.eph import eph_fd

        atoms = [
            (self.mol.symbol(i), tuple(self.mol.xyz(i)[j] for j in range(3)))
            for i in range(self.mol.natom())]
        mol = gto.M(
            atom=atoms,
            basis=self.basis,
            unit="Bohr",
            charge=int(self.mol.molecular_charge()),
            spin=int(self.mol.multiplicity() - 1),
            symmetry=False,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1.0e-12
        mf.conv_tol_grad = 1.0e-8
        mf.max_cycle = 100
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("PySCF compatibility RHF did not converge")
        mats, omega = eph_fd.kernel(
            mf,
            disp=self.fd_step,
            mo_rep=True,
            cutoff_frequency=self.cutoff_frequency,
            keep_imag_frequency=False,
        )
        # Psi4's Gaussian solid-harmonic component order differs from
        # PySCF's order.  Reorder the Psi4 coefficients before using their
        # cross-code MO overlap to rotate g into the Psi4 canonical MO gauge.
        component_order = {
            0: (0,),
            1: (1, 2, 0),       # Psi4 (z,x,y) -> PySCF (x,y,z)
            2: (4, 2, 0, 1, 3), # Psi4 -> PySCF (xy,yz,z2,xz,x2-y2)
        }
        psi4_rows_in_pyscf_order = []
        for shell_index in range(self.reference_basis.nshell()):
            shell = self.reference_basis.shell(shell_index)
            angular_momentum = int(shell.am)
            if angular_momentum not in component_order:
                raise NotImplementedError(
                    "Psi4/PySCF MO gauge alignment currently supports s, p, "
                    f"and d shells only; found l={angular_momentum}")
            order = component_order[angular_momentum]
            if len(order) != int(shell.nfunction):
                raise NotImplementedError(
                    "Cartesian high-angular-momentum AO gauge alignment is "
                    "not implemented")
            start = int(shell.function_index)
            psi4_rows_in_pyscf_order.extend(start + item for item in order)
        coeff_psi4 = self._array(self.wfn.Ca())[
            np.asarray(psi4_rows_in_pyscf_order), :]
        overlap = mf.get_ovlp()
        metric_error = np.linalg.norm(
            coeff_psi4.T.conj() @ overlap @ coeff_psi4
            - np.eye(coeff_psi4.shape[1]))
        mo_rotation = mf.mo_coeff.T.conj() @ overlap @ coeff_psi4
        rotation_error = np.linalg.norm(
            mo_rotation.T.conj() @ mo_rotation
            - np.eye(mo_rotation.shape[1]))
        if metric_error > 1.0e-5 or rotation_error > 1.0e-5:
            raise RuntimeError(
                "Could not align PySCF and Psi4 MO gauges; verify AO ordering "
                f"and basis conventions (metric errors {metric_error:.3e}, "
                f"{rotation_error:.3e})")
        mats_psi4_mo = np.einsum(
            "pi,Jpq,qj->Jij", mo_rotation.conj(), mats, mo_rotation,
            optimize=True)
        return mats_psi4_mo, omega

    def run(self, vib_results):
        if self.engine == "pyscf":
            log.info(
                "Computing Pulay-correct EPH with the PySCF compatibility "
                "engine at the Psi4 geometry")
            return self._run_pyscf()
        if self.engine != "psi4_approx":
            raise ValueError(
                f"Unknown Psi4 EPH engine {self.engine!r}; expected 'pyscf' "
                "or 'psi4_approx'")
        log.warning(
            "Using psi4_approx EPH without the full AO Pulay derivative; "
            "matrix elements are not expected to match PySCF")
        freq = np.asarray(vib_results["freq_wavenumber"]).real
        modes = np.asarray(vib_results["norm_mode"], dtype=float)
        if modes.ndim != 3 or modes.shape[1:] != (self.mol.natom(), 3):
            raise ValueError("Psi4 normal modes must have shape (nmodes, natom, 3)")
        keep = np.abs(freq) >= self.cutoff_frequency
        keep &= freq > 0
        freq_au = freq[keep] / HARTREE_TO_WAVENUMBER
        modes = modes[keep]
        masses = np.asarray([self.mol.mass(i) for i in range(self.mol.natom())])
        # Normalize every Cartesian mode in the same mass metric used by
        # PySCF: sum_{A,alpha} M_A L[A,alpha]^2 = 1.  Psi4 frequency
        # metadata and the Hessian fallback do not share one guaranteed
        # eigenvector normalization, so normalize explicitly here.
        mass_au = masses * AMU_TO_ELECTRON_MASS
        mass_norm = np.sqrt(np.sum(
            (mass_au[None, :, None] * modes * modes), axis=(1, 2)))
        modes = modes / mass_norm[:, None, None]
        coords = np.asarray([[self.mol.xyz(i)[j] for j in range(3)]
                             for i in range(self.mol.natom())])
        symbols = [self.mol.symbol(i) for i in range(self.mol.natom())]
        c0 = self._array(self.wfn.Ca())
        # First differentiate in each Cartesian nuclear coordinate, as in
        # pyscf.eph.eph_fd, then project the Cartesian derivatives into normal
        # modes with the mass- and zero-point-weighted eigenvectors.
        nat = self.mol.natom()
        cart_deriv = []
        for atom in range(nat):
            for axis in range(3):
                delta = np.zeros_like(coords)
                delta[atom, axis] = self.fd_step / 2.0
                fp = self._displaced_fock(coords + delta, symbols)
                fm = self._displaced_fock(coords - delta, symbols)
                derivative_ao = (fp - fm) / self.fd_step
                cart_deriv.append(c0.T.conj() @ derivative_ao @ c0)
        cart_deriv = np.asarray(cart_deriv)
        # Psi4's mode array is (mode, atom, xyz); PySCF's projection vector is
        # flattened in atom-major Cartesian order.
        mode_vec = modes.transpose(1, 2, 0).reshape(3 * nat, len(freq_au))
        # The modes are now mass-normalized; only the zero-point factor
        # 1/sqrt(2 omega) remains in the EPH coordinate transformation.
        mode_vec = mode_vec / np.sqrt(2.0 * freq_au[None, :])
        mats = np.einsum("xJ,xpq->Jpq", mode_vec, cart_deriv)
        return mats, freq_au
