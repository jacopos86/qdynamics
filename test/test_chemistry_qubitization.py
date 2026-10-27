"""End-to-end encoding checks against occupation-basis Hamiltonians."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.chemistry.pyscf.read_write_pyscf_results import write_matrix_elements_h5, write_eph_h5
from src.chemistry_qubitization import load_integrals, build_hamiltonian, save_hamiltonian, load_hamiltonian


def sources(tmp_path, *, eph=False):
    electronic = tmp_path / "job_ele.h5"
    h2 = np.zeros((2,) * 4)
    for a in (0, 1):
        for b in (0, 1):
            h2[a, a, b, b] = 0.2
    meta = dict(nelec=2, spin=0, method="rhf", basis="synthetic", mol_str="test", converged=True)
    write_matrix_elements_h5(electronic, np.diag([0.7, 0.7]), h2, 0.1, meta)
    if eph:
        write_eph_h5(tmp_path / "job_eph.h5", np.array([[[0.3]]]), np.array([0.4]), {**meta, "eph_basis": "MO"})
    return electronic


def dense(model):
    matrices = {"e": np.eye(2), "x": np.array([[0, 1], [1, 0]]),
                "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1])}
    result = np.zeros((1 << model.n_qubits,) * 2, dtype=complex)
    for term in model.operator.return_polynomial():
        matrix = np.array([[1]], dtype=complex)
        for symbol in term.pw2strng():
            matrix = np.kron(matrix, matrices[symbol])
        result += term.p_coeff * matrix
    return result


def test_electronic_roundtrip(tmp_path):
    sources(tmp_path)
    model = build_hamiltonian(load_integrals([tmp_path]))
    assert model.n_qubits == 2
    np.testing.assert_allclose(dense(model), np.diag([0.1, 0.8, 0.8, 1.7]), atol=1e-12)
    output = tmp_path / "hamiltonian.json"
    save_hamiltonian(model, output)
    restored = load_hamiltonian(output)
    np.testing.assert_allclose(dense(restored), dense(model))
    assert restored.metadata == model.metadata
    with pytest.raises(FileExistsError):
        save_hamiltonian(model, output)


def test_package_does_not_import_legacy_qubitization():
    root = Path(__file__).resolve().parents[1]
    script = """
import builtins
original_import = builtins.__import__
def reject_legacy(name, *args, **kwargs):
    if name == 'src.qubitization' or name.startswith('src.qubitization.'):
        raise AssertionError(f'legacy import attempted: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = reject_legacy
import src.chemistry_qubitization
from src.chemistry_qubitization import PauliPolynomial, PauliTerm
assert PauliPolynomial.__module__ == 'src.chemistry_qubitization.pauli'
assert PauliTerm.__module__ == 'src.chemistry_qubitization.pauli'
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("encoding", ["binary", "unary"])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_vibronic_physical_matrix(tmp_path, encoding, cutoff):
    sources(tmp_path, eph=True)
    model = build_hamiltonian(load_integrals([tmp_path]), cutoffs=[cutoff], encoding=encoding)
    boson_codes = list(range(cutoff + 1)) if encoding == "binary" else [1 << n for n in range(cutoff + 1)]
    indices = [code * 4 + f for code in boson_codes for f in range(4)]
    b = np.diag(np.sqrt(np.arange(1, cutoff+1)), 1)
    expected = (np.kron(np.eye(cutoff+1), np.diag([0.1, 0.8, 0.8, 1.7]))
                + 0.4 * np.kron(np.diag(np.arange(cutoff+1)+0.5), np.eye(4))
                + 0.3 * np.kron(b+b.T, np.diag([0, 1, 1, 2])))
    actual = dense(model)
    np.testing.assert_allclose(actual[np.ix_(indices, indices)], expected, atol=1e-12)
    illegal = sorted(set(range(len(actual))) - set(indices))
    np.testing.assert_allclose(actual[np.ix_(illegal, indices)], 0, atol=1e-12)


def test_multiorbital_fermion_mapping(tmp_path):
    """Independent Fock-space ladder matrices check ERI indices and JW signs."""
    rng = np.random.default_rng(17)
    h1 = np.array([[-0.8, 0.17], [0.17, -0.3]])
    raw = rng.normal(size=(2,) * 4)
    eri = raw + raw.swapaxes(0, 1)
    eri = eri + eri.swapaxes(2, 3)
    eri = (eri + eri.transpose(2, 3, 0, 1)) / 8
    spin1, spin2 = np.zeros((4, 4)), np.zeros((4,) * 4)
    for a in (0, 1):
        spin1[a::2, a::2] = h1
        for b in (0, 1):
            spin2[a::2, a::2, b::2, b::2] = eri
    write_matrix_elements_h5(tmp_path / "molecule.h5", spin1, spin2, 0.27, dict(nelec=2, spin=0))
    model = build_hamiltonian(load_integrals([tmp_path]))
    annihilators = []
    for p in range(4):
        operator = np.zeros((16, 16))
        for state in range(16):
            if state & (1 << p):
                operator[state ^ (1 << p), state] = (-1) ** ((state & ((1 << p)-1)).bit_count())
        annihilators.append(operator)
    expected = 0.27 * np.eye(16)
    for p in range(4):
        for q in range(4):
            if p//2 == q//2:
                expected += h1[p % 2, q % 2] * annihilators[p].T @ annihilators[q]
            for r in range(4):
                for s in range(4):
                    if p//2 == r//2 and q//2 == s//2:
                        expected += 0.5 * eri[p % 2, r % 2, q % 2, s % 2] * annihilators[p].T @ annihilators[q].T @ annihilators[s] @ annihilators[r]
    np.testing.assert_allclose(dense(model), expected, atol=1e-12)


def test_cli_module_and_script(tmp_path):
    sources(tmp_path, eph=True)
    root = Path(__file__).resolve().parents[1]
    for invocation in (["-m", "src.chemistry_qubitization"], [str(root / "src/chemistry_qubitization/__main__.py")]):
        output = tmp_path / ("module.json" if invocation[0] == "-m" else "script.json")
        result = subprocess.run([sys.executable, *invocation, str(tmp_path), "--cutoffs", "1", "-o", str(output)], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())["n_qubits"] == 3


def test_ambiguity_and_bad_inputs(tmp_path):
    electronic = sources(tmp_path, eph=True)
    with pytest.raises(ValueError, match="cutoff"):
        build_hamiltonian(load_integrals([tmp_path]))
    with pytest.raises(ValueError, match="positive integer"):
        build_hamiltonian(load_integrals([tmp_path]), cutoffs=[0])
    with h5py.File(tmp_path / "job_eph.h5", "a") as handle:
        handle["metadata"].attrs["basis"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        load_integrals([tmp_path])
    with h5py.File(electronic, "a") as handle:
        handle["h1"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        load_integrals([electronic])


def test_multiple_calculations_require_explicit_selection(tmp_path):
    first = sources(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    second = sources(other)
    with pytest.raises(ValueError, match="Select files explicitly"):
        load_integrals([first, second])
    assert build_hamiltonian(load_integrals([first])).n_qubits == 2


def test_multimode_registers_and_cutoffs(tmp_path):
    sources(tmp_path, eph=True)
    with h5py.File(tmp_path / "job_eph.h5", "a") as handle:
        del handle["eph_mat"]
        del handle["omega"]
        handle["eph_mat"] = np.array([[[0.3]], [[-0.2]]])
        handle["omega"] = np.array([0.4, 0.6])
    integrals = load_integrals([tmp_path])
    model = build_hamiltonian(integrals, cutoffs=[1, 2])
    expected = np.kron(np.eye(8), np.diag([0.1, 0.8, 0.8, 1.7]))
    b1 = np.array([[0, 1], [0, 0]])
    b2 = np.zeros((4, 4))
    b2[0, 1], b2[1, 2] = 1, np.sqrt(2)
    expected += 0.4 * np.kron(np.kron(np.eye(4), b1.T @ b1 + 0.5*np.eye(2)), np.eye(4))
    expected += 0.6 * np.kron(np.kron(b2.T @ b2 + 0.5*np.eye(4), np.eye(2)), np.eye(4))
    expected += np.kron(0.3*np.kron(np.eye(4), b1+b1.T) - 0.2*np.kron(b2+b2.T, np.eye(2)), np.diag([0, 1, 1, 2]))
    np.testing.assert_allclose(dense(model), expected, atol=1e-12)
    assert build_hamiltonian(integrals, cutoffs=[1]).cutoffs == (1, 1)
    with pytest.raises(ValueError, match="per phonon mode"):
        build_hamiltonian(integrals, cutoffs=[1, 2, 3])
