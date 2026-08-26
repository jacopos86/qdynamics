"""Numerical regression for native Psi4 Pulay-correct FD EPH."""

import numpy as np
import pytest

psi4 = pytest.importorskip("psi4")
pytest.importorskip("pyscf")
from pyscf import gto, scf
from pyscf.eph import eph_fd

from src.chemistry.psi4.electron_phonon_fd_solver import (
    FiniteDifferenceElectronPhononSolver,
    HARTREE_TO_WAVENUMBER,
)
from src.chemistry.pyscf.electron_phonon_solver import (
    standardize_pyscf_eph,
)


class _MoleculeWrapper:
    def __init__(self, geometry):
        self.geometry = geometry


class _WavefunctionWrapper:
    def __init__(self, geometry, wavefunction):
        self.mol_struct = _MoleculeWrapper(geometry)
        self.wfn = wavefunction

    def Ca(self):
        return self.wfn.Ca()


def test_h2_psi4_fd_matches_pyscf_pulay_fd(tmp_path):
    psi4.core.set_output_file(str(tmp_path / "psi4-eph-regression.out"), False)
    psi4.set_options({
        "basis": "sto-3g",
        "reference": "rhf",
        "scf_type": "pk",
        "e_convergence": 1.0e-12,
        "d_convergence": 1.0e-10,
    })
    molecule = psi4.geometry("""
        0 1
        H 0 0 -0.7
        H 0 0  0.7
        units bohr
        symmetry c1
        no_reorient
        no_com
    """)
    _, core_wfn = psi4.energy("scf", molecule=molecule, return_wfn=True)
    wrapped = _WavefunctionWrapper(molecule, core_wfn)
    mode = np.zeros((1, 2, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0
    vibration = {
        "freq_wavenumber": np.array([5028.0]),
        "norm_mode": mode,
    }

    native, native_omega = FiniteDifferenceElectronPhononSolver(
        wrapped, "scf", fd_step=0.002, basis="sto-3g",
        engine="psi4_fd").run(vibration)
    pyscf_mol = gto.M(
        atom="H 0 0 -0.7; H 0 0 0.7", unit="Bohr", basis="sto-3g",
        spin=0, charge=0, verbose=0)
    pyscf_mol.nucprop = {
        1: {"mass": molecule.mass(0)}, 2: {"mass": molecule.mass(1)}}
    pyscf_mf = scf.RHF(pyscf_mol)
    pyscf_mf.conv_tol = 1.0e-12
    pyscf_mf.conv_tol_grad = 1.0e-8
    pyscf_mf.kernel()
    reference, reference_omega = eph_fd.kernel(
        pyscf_mf, disp=0.002, mo_rep=True, cutoff_frequency=80.0,
        keep_imag_frequency=False)
    reference, reference_omega = standardize_pyscf_eph(
        reference, reference_omega)

    expected_omega = 5028.0 / HARTREE_TO_WAVENUMBER
    assert native_omega == pytest.approx([expected_omega], rel=1.0e-12)
    assert np.linalg.norm(native) == pytest.approx(
        np.linalg.norm(reference), rel=1.0e-4)
    reference_mode = reference[
        np.argmax(np.linalg.norm(reference, axis=(1, 2)))]
    error = min(
        np.linalg.norm(native[0] - reference_mode),
        np.linalg.norm(native[0] + reference_mode),
    )
    assert error / np.linalg.norm(reference_mode) < 1.0e-4
