"""Finite-difference Psi4 mean-field electron--phonon coupling."""
import logging
import numpy as np
import psi4

log = logging.getLogger(__name__)
HARTREE_TO_WAVENUMBER = 219474.6313705
AMU_TO_ELECTRON_MASS = 1822.888486209


class FiniteDifferenceElectronPhononSolver:
    """Build g from displaced Psi4 Fock matrix elements.

    The equilibrium MO coefficients are used to represent every displaced
    Fock matrix in one common basis.  ``norm_mode`` is expected in the
    (mode, atom, xyz) convention returned by :class:`VibrationalSolver`.
    """
    def __init__(self, wavefunction, method, fd_step=0.001,
                 cutoff_frequency=80.0):
        self.wfn = wavefunction
        self.mol = wavefunction.mol_struct.geometry
        self.method = method
        self.fd_step = float(fd_step)
        self.cutoff_frequency = float(cutoff_frequency)

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
                        for s, r in zip(symbols, coords)] + ["units bohr", "no_reorient", "no_com"]
        ))
        _, wfn = psi4.energy(self.method, molecule=mol, return_wfn=True)
        # PySCF's eph_fd differentiates the self-consistent electronic
        # potential (Fock minus kinetic operator), not the bare full Fock
        # matrix.  Use the same object here.
        fock = self._array(wfn.Fa())
        kinetic = self._array(wfn.mintshelper().ao_kinetic())
        return fock - kinetic

    def run(self, vib_results):
        freq = np.asarray(vib_results["freq_wavenumber"]).real
        modes = np.asarray(vib_results["norm_mode"], dtype=float)
        if modes.ndim != 3 or modes.shape[1:] != (self.mol.natom(), 3):
            raise ValueError("Psi4 normal modes must have shape (nmodes, natom, 3)")
        keep = np.abs(freq) >= self.cutoff_frequency
        keep &= freq > 0
        freq_au = freq[keep] / HARTREE_TO_WAVENUMBER
        modes = modes[keep]
        masses = np.asarray([self.mol.mass(i) for i in range(self.mol.natom())])
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
        mass_vec = np.repeat(masses * AMU_TO_ELECTRON_MASS, 3)
        mode_vec = mode_vec / np.sqrt(2.0 * mass_vec[:, None] * freq_au[None, :])
        mats = np.einsum("xJ,xpq->Jpq", mode_vec, cart_deriv)
        return mats, freq_au
