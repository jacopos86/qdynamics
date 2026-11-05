"""Our vibronic models against Doron Wu's qubitization conventions and JSON schema (qdynamics)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.chemistry.molecular_vibronic import (  # noqa: E402
    ActiveSpace,
    ActiveTensors,
    LinearCoupling,
    MolecularGeometry,
    NormalMode,
    assemble_vibronic_model,
    parse_geometry,
)
from src.quantum.chemistry.qdynamics_interop import (  # noqa: E402
    QDYNAMICS_SCHEMA,
    builder_model_in_qdynamics_conventions,
    compare_pauli_polynomials,
    hartree_fock_vacuum_energy,
    load_qdynamics_qubit_hamiltonian,
    physical_sector_energies,
    qdynamics_payload_from_model,
    qdynamics_qubit_hamiltonian_from_payload,
    save_qdynamics_qubit_hamiltonian,
)
from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix  # noqa: E402

PAULI = {"e": np.eye(2), "x": np.array([[0, 1], [1, 0]]), "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1.0, -1.0])}


def _dense_from_words(terms, nq):
    """Dense matrix from (word, coefficient) pairs in q_(n-1)...q_0 display order (his test's `dense`)."""
    out = np.zeros((1 << nq, 1 << nq), dtype=complex)
    for word, coefficient in terms:
        matrix = np.array([[1.0]], dtype=complex)
        for letter in word:
            matrix = np.kron(matrix, PAULI[letter])
        out += coefficient * matrix
    return out


def _his_tiny_molecule(cutoff, encoding):
    """The synthetic molecule of his test_chemistry_qubitization.py: one spatial orbital, two electrons,
    h = 0.7, (00|00) = 0.2, E_nuc = 0.1, one mode omega = 0.4 with g = 0.3 (zero-point factor included),
    assembled with OUR builder: C per unit Q = sqrt(2 omega) g."""
    tensors = ActiveTensors(0.1, np.array([[0.7]]), np.full((1, 1, 1, 1), 0.2))
    space = ActiveSpace((), (0,), (), 1, 1, "full")
    mode = NormalMode(0, 0.4, np.zeros((2, 3)), "mode_0")
    coupling = LinearCoupling(0, "mean_field_pyscf_eph", 0.0, np.array([[0.3 * np.sqrt(2 * 0.4)]]), None, 0.0)
    geometry = MolecularGeometry(("H", "H"), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    return assemble_vibronic_model(geometry=geometry, basis="synthetic", active_space=space, active_tensors=tensors,
                                   modes=(mode,), couplings=(coupling,), cutoffs=cutoff, encoding=encoding)


def _his_expected_matrix(cutoff):
    """His test_vibronic_physical_matrix expectation on the physical block, in his index order."""
    b = np.diag(np.sqrt(np.arange(1, cutoff + 1)), 1)
    return (np.kron(np.eye(cutoff + 1), np.diag([0.1, 0.8, 0.8, 1.7]))
            + 0.4 * np.kron(np.diag(np.arange(cutoff + 1) + 0.5), np.eye(4))
            + 0.3 * np.kron(b + b.T, np.diag([0.0, 1.0, 1.0, 2.0])))


@pytest.mark.parametrize("encoding", ["binary", "unary"])
@pytest.mark.parametrize("cutoff", [1, 2])
def test_our_assembly_reproduces_his_reference_matrix(cutoff, encoding):
    model = _his_tiny_molecule(cutoff, encoding)
    actual = np.asarray(hamiltonian_matrix(model.h_vibronic))
    boson_codes = list(range(cutoff + 1)) if encoding == "binary" else [1 << n for n in range(cutoff + 1)]
    indices = [code * 4 + f for code in boson_codes for f in range(4)]   # fermion qubits are the low bits, as in his code
    np.testing.assert_allclose(actual[np.ix_(indices, indices)], _his_expected_matrix(cutoff), atol=1e-12)
    illegal = sorted(set(range(len(actual))) - set(indices))
    np.testing.assert_allclose(actual[np.ix_(illegal, indices)], 0.0, atol=1e-12)
    # The same statements through the interop helpers.
    assert hartree_fock_vacuum_energy(model) == pytest.approx(1.7 + 0.2, abs=1e-12)   # E(2 electrons) + omega/2, coupling invisible
    assert physical_sector_energies(model)[0] == pytest.approx(np.linalg.eigvalsh(_his_expected_matrix(cutoff)[np.ix_(
        [3 + 4 * k for k in range(cutoff + 1)], [3 + 4 * k for k in range(cutoff + 1)])])[0], abs=1e-12)


def test_payload_round_trip_and_his_loader_rules(tmp_path):
    model = _his_tiny_molecule(2, "binary")
    payload = qdynamics_payload_from_model(model, source_files=["synthetic"], metadata={"case": "tiny"})
    assert payload["schema"] == QDYNAMICS_SCHEMA and payload["coupling_model"] == "mean_field_potential_eph"
    assert (payload["spin_ordering"], payload["pauli_ordering"], payload["energy_units"]) == ("blocked", "q_(n-1)...q_0", "hartree")
    assert payload["n_qubits"] == 4 and payload["n_fermion_qubits"] == 2 and payload["num_particles"] == [1, 1]
    assert payload["mode_cutoffs"] == [2] and payload["boson_encoding"] == "binary" and payload["omega_hartree"] == [0.4]
    words = [t["word"] for t in payload["terms"]]
    assert words == sorted(words) and all(len(w) == 4 and set(w) <= set("exyz") for w in words)
    assert all(len(t["coefficient"]) == 2 and np.isfinite(t["coefficient"]).all() for t in payload["terms"])
    path = save_qdynamics_qubit_hamiltonian(model, tmp_path / "tiny.json", source_files=["synthetic"], metadata={"case": "tiny"})
    with pytest.raises(FileExistsError):
        save_qdynamics_qubit_hamiltonian(model, path)
    loaded = load_qdynamics_qubit_hamiltonian(path)
    assert loaded.n_qubits == 4 and loaded.num_particles == (1, 1) and loaded.mode_cutoffs == (2,) and loaded.encoding == "binary"
    assert loaded.layout.n_total_qubits == 4 and loaded.layout.boson_modes[0].n_ph_max == 2 and loaded.metadata["case"] == "tiny"
    report = compare_pauli_polynomials(loaded.operator, model.h_vibronic)
    assert report["equal_within_atol"] and report["n_only_in_a"] == 0 and report["n_only_in_b"] == 0
    assert hartree_fock_vacuum_energy(loaded) == pytest.approx(hartree_fock_vacuum_energy(model), abs=1e-12)
    assert physical_sector_energies(loaded, n_low=2) == pytest.approx(physical_sector_energies(model, n_low=2), abs=1e-12)
    # A file in his writer's exact shape (word + [re, im]) loads into the same operator.
    his_shape = json.loads(path.read_text())
    his_shape["terms"] = [{"word": t["word"], "coefficient": t["coefficient"]} for t in his_shape["terms"]]
    assert compare_pauli_polynomials(qdynamics_qubit_hamiltonian_from_payload(his_shape).operator, model.h_vibronic)["equal_within_atol"]
    # His loader's refusals, mirrored.
    for key, value in (("schema", "other"), ("spin_ordering", "interleaved"), ("pauli_ordering", "q_0...q_(n-1)"), ("energy_units", "eV")):
        broken = dict(his_shape, **{key: value})
        with pytest.raises(ValueError):
            qdynamics_qubit_hamiltonian_from_payload(broken)
    broken = dict(his_shape, terms=[{"word": "exyz", "coefficient": [1.0, 0.0]}, {"word": "exy", "coefficient": [1.0, 0.0]}])
    with pytest.raises(ValueError, match="invalid Pauli term"):
        qdynamics_qubit_hamiltonian_from_payload(broken)
    with pytest.raises(ValueError, match="disagrees"):
        qdynamics_qubit_hamiltonian_from_payload(dict(his_shape, n_qubits=5))


def test_compare_reports_differences():
    a = _his_tiny_molecule(1, "binary").h_vibronic
    b = _his_tiny_molecule(2, "binary").h_vibronic
    report = compare_pauli_polynomials(a, b)
    assert not report["equal_within_atol"] and report["n_only_in_b"] > 0 and report["n_terms_a"] < report["n_terms_b"]


def test_his_benchmark_ground_state_is_the_reference_energy_h2():
    pytest.importorskip("pyscf")
    # RHF/STO-3G optimum of H2 (0.712 A); his table reports -1.1050156404184244 Ha for H2.
    model, computation = builder_model_in_qdynamics_conventions(parse_geometry("H 0 0 0; H 0 0 0.7120"), cutoffs=1)
    assert model.coupling_model == "mean_field_pyscf_eph" and model.provenance["reference_force_counter_term"] is False
    reference = computation.center.hf_energy + 0.5 * model.modes[0].frequency_hartree
    assert hartree_fock_vacuum_energy(model) == pytest.approx(reference, abs=1e-10)
    assert reference == pytest.approx(-1.1050156404184244, abs=2e-5)
    # The actual ground state of his operator lies far below: the uncompensated mean-field force
    # displaces the oscillator and the electrons correlate.
    ground = physical_sector_energies(model)[0]
    assert ground < reference - 0.05
    payload = qdynamics_payload_from_model(model)
    assert payload["n_qubits"] == 5 and payload["coupling_model"] == "mean_field_potential_eph"
