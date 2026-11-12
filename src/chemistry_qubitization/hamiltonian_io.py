"""Read and write Pauli Hamiltonian JSON files with source and encoding metadata."""
import json
from pathlib import Path
import numpy as np
from .data import QubitHamiltonian
from .pauli import PauliPolynomial, PauliTerm


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_hamiltonian(model, path):
    payload = dict(schema="qdynamics_qubit_hamiltonian_v1", energy_units="hartree",
                   spin_ordering="blocked", pauli_ordering="q_(n-1)...q_0",
                   coupling_model="mean_field_potential_eph" if model.cutoffs else None,
                   n_qubits=model.n_qubits, n_fermion_qubits=model.n_fermion_qubits,
                   num_particles=model.particles, mode_cutoffs=model.cutoffs,
                   boson_encoding=model.encoding, omega_hartree=model.frequencies,
                   source_files=model.source_files, metadata=model.metadata,
                   terms=[dict(word=t.pw2strng(), coefficient=[complex(t.p_coeff).real, complex(t.p_coeff).imag])
                          for t in sorted(model.operator.return_polynomial(), key=lambda t: t.pw2strng())])
    text = json.dumps(payload, indent=2, default=_json_value, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        handle.write(text)


def load_hamiltonian(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["schema"] != "qdynamics_qubit_hamiltonian_v1" or payload["spin_ordering"] != "blocked" or payload["pauli_ordering"] != "q_(n-1)...q_0" or payload["energy_units"] != "hartree":
        raise ValueError("Unsupported Hamiltonian schema or conventions")
    nq = payload["n_qubits"]
    terms = []
    for term in payload["terms"]:
        word, coefficient = term["word"], complex(*term["coefficient"])
        if len(word) != nq or set(word) - set("exyz") or not np.isfinite(coefficient):
            raise ValueError("Invalid Pauli term")
        terms.append(PauliTerm(nq, ps=word, pc=coefficient))
    return QubitHamiltonian(PauliPolynomial("JW", terms), nq, payload["n_fermion_qubits"],
                            tuple(payload["num_particles"]), tuple(payload["mode_cutoffs"]),
                            payload["boson_encoding"], tuple(payload["omega_hartree"]),
                            tuple(payload["source_files"]), payload["metadata"])
