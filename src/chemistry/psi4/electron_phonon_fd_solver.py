"""Finite-difference Psi4 mean-field electron--phonon coupling."""
import logging
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import numpy as np
import psi4

log = logging.getLogger(__name__)
HARTREE_TO_WAVENUMBER = 219474.6313705
AMU_TO_ELECTRON_MASS = 1822.888486209


class FiniteDifferenceElectronPhononSolver:
    """Build Pulay-correct finite-difference EPH matrices with Psi4.

    The native ``psi4_fd`` engine differences displaced Psi4 SCF operators in
    their atom-labelled AO representation.  It removes basis-centre motion
    using a fixed-nuclei finite difference for nuclear attraction and a small
    Psi4 integral plugin for the single-leg two-electron derivative contracted
    with the reference density.  This gives the same bra/ket Pulay correction
    used by an atom-labelled moving-AO finite difference.  This production
    implementation has no PySCF dependency.
    """
    def __init__(self, wavefunction, method, fd_step=0.001,
                 cutoff_frequency=80.0, basis=None, engine="psi4_fd"):
        self.wfn = wavefunction
        self.mol = wavefunction.mol_struct.geometry
        self.method = method
        self.fd_step = float(fd_step)
        self.cutoff_frequency = float(cutoff_frequency)
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

    def _molecule(self, coords, symbols):
        """Build a c1 Psi4 molecule without translating or reorienting it."""
        mol = psi4.geometry("\n".join(
            [f"{int(self.mol.molecular_charge())} "
             f"{int(self.mol.multiplicity())}"]
            + [f"{s} {r[0]:.14f} {r[1]:.14f} {r[2]:.14f}"
                        for s, r in zip(symbols, coords)]
            + ["units bohr", "symmetry c1", "no_reorient", "no_com"]
        ))
        return mol

    def _displaced_operator(self, coords, symbols):
        """Return displaced ``F - T`` in its atom-labelled AO basis."""
        mol = self._molecule(coords, symbols)
        _, wfn = psi4.energy(self.method, molecule=mol, return_wfn=True)
        if wfn.nirrep() != 1:
            raise RuntimeError("Displaced Psi4 EPH calculations require symmetry c1")
        # Match the electronic operator differentiated by PySCF: F - T.
        fock = self._array(wfn.Fa())
        kinetic = self._array(
            psi4.core.MintsHelper(wfn.basisset()).ao_kinetic())
        return fock - kinetic, wfn.basisset()

    @staticmethod
    def _mean_field_potential(nuclear_potential, eri, density_a, density_b):
        """Build the alpha-spin ``V_nuc + J[D_a+D_b] - K[D_a]``."""
        coulomb = np.einsum(
            "mnkl,kl->mn", eri, density_a + density_b, optimize=True)
        exchange = np.einsum(
            "mknl,kl->mn", eri, density_a, optimize=True)
        return nuclear_potential + coulomb - exchange

    def _basis_only_nuclear_potential(self, basis, reference_coords):
        """Evaluate nuclear attraction in a moved AO basis at fixed nuclei."""
        external = psi4.core.ExternalPotential()
        for atom, coord in enumerate(reference_coords):
            external.addCharge(
                float(self.mol.Z(atom)),
                float(coord[0]), float(coord[1]), float(coord[2]))
        nuclear_potential = self._array(external.computePotentialMatrix(basis))
        return nuclear_potential

    @staticmethod
    def _plugin_path():
        override = os.environ.get("QDYNAMICS_PSI4_EPH_PLUGIN")
        if override:
            return Path(override)
        source = Path(__file__).with_name("eph_deriv_plugin")
        build = Path(tempfile.gettempdir()) / (
            f"qdynamics_psi4_eph_deriv_{psi4.__version__}")
        library = build / "qdynamics_eph_deriv.so"
        if library.exists() and library.stat().st_mtime >= max(
                path.stat().st_mtime for path in source.iterdir()):
            return library
        python_root = Path(sysconfig.get_config_var("prefix"))
        library_name = sysconfig.get_config_var("LDLIBRARY") or ""
        python_library = python_root / "lib" / library_name
        if not python_library.exists():
            python_library = python_root / "lib" / (
                f"libpython{sysconfig.get_python_version()}.dylib")
        pybind11_dir = (
            Path(sysconfig.get_path("purelib"))
            / "pybind11" / "share" / "cmake" / "pybind11")
        command = [
            "cmake", "-S", str(source), "-B", str(build),
            f"-DCMAKE_PREFIX_PATH={python_root}",
            f"-DPython_ROOT_DIR={python_root}",
            f"-DPython_EXECUTABLE={Path(os.sys.executable)}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_config_var('INCLUDEPY')}",
            f"-DPython_LIBRARY={python_library}",
            f"-Dpybind11_DIR={pybind11_dir}",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        subprocess.run(
            ["cmake", "--build", str(build), "-j2"],
            check=True, capture_output=True, text=True)
        return library

    def _two_electron_first_leg(self, atom):
        """Return ``d(J-K)/dR`` acting on the first AO leg only."""
        plugin = str(self._plugin_path())
        psi4.core.set_local_option("QDYNAMICS_EPH_DERIV", "ATOM", atom)
        result_wfn = psi4.core.plugin(plugin, self.wfn.wfn)
        return np.asarray([
            self._array(result_wfn.array_variable(
                f"QDYNAMICS EPH TWO ELECTRON LEG {xyz}"))
            for xyz in range(3)
        ])

    def run(self, vib_results):
        if self.engine not in ("psi4", "psi4_fd"):
            raise ValueError(
                f"Unknown Psi4 EPH engine {self.engine!r}; expected "
                "'psi4_fd'")
        if str(self.method).lower() not in ("scf", "hf", "rhf"):
            raise NotImplementedError(
                "Native Psi4 FD EPH currently supports RHF/SCF only")
        if self.wfn.wfn.nalpha() != self.wfn.wfn.nbeta():
            raise NotImplementedError(
                "Native Psi4 FD EPH currently supports closed shells only")
        log.info(
            "Computing native Pulay-correct Psi4 finite-difference EPH")
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
        # Difference the total self-consistent potential in the moving AO
        # basis, then subtract the derivative caused solely by moving the AO
        # centres at fixed nuclei and fixed reference density.  The remainder
        # is the Pulay-correct Cartesian electronic-potential derivative.
        nat = self.mol.natom()
        cart_deriv = []
        # Loading once registers the plugin-local ATOM option.  Do this for
        # every run because callers may have reset Psi4's global option state.
        psi4.core.plugin(str(self._plugin_path()), self.wfn.wfn)
        for atom in range(nat):
            two_electron_leg = self._two_electron_first_leg(atom)
            for axis in range(3):
                delta = np.zeros_like(coords)
                delta[atom, axis] = self.fd_step / 2.0
                fp, basis_p = self._displaced_operator(
                    coords + delta, symbols)
                fm, basis_m = self._displaced_operator(
                    coords - delta, symbols)
                basis_fp = self._basis_only_nuclear_potential(basis_p, coords)
                basis_fm = self._basis_only_nuclear_potential(basis_m, coords)
                derivative_ao = (
                    (fp - fm) - (basis_fp - basis_fm)) / self.fd_step
                derivative_ao -= (
                    two_electron_leg[axis]
                    + two_electron_leg[axis].T.conj())
                derivative_mo = c0.T.conj() @ derivative_ao @ c0
                cart_deriv.append(
                    0.5 * (derivative_mo + derivative_mo.T.conj()))
        cart_deriv = np.asarray(cart_deriv)
        # Psi4's mode array is (mode, atom, xyz); PySCF's projection vector is
        # flattened in atom-major Cartesian order.
        mode_vec = modes.transpose(1, 2, 0).reshape(3 * nat, len(freq_au))
        # The modes are now mass-normalized; only the zero-point factor
        # 1/sqrt(2 omega) remains in the EPH coordinate transformation.
        mode_vec = mode_vec / np.sqrt(2.0 * freq_au[None, :])
        mats = np.einsum("xJ,xpq->Jpq", mode_vec, cart_deriv)
        return mats, freq_au
