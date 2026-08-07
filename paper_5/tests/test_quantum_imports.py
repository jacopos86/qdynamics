from __future__ import annotations

from paper5.quantum_imports import PauliPolynomial, PauliTerm, jw_annihilation, jw_creation


def test_jw_ladder_uses_parent_repo_pauli_convention() -> None:
    creation = jw_creation(n_qubits=3, mode=1)
    annihilation = jw_annihilation(n_qubits=3, mode=1)

    assert isinstance(creation, PauliPolynomial)
    assert isinstance(annihilation, PauliPolynomial)
    assert {term.pw2strng() for term in creation.return_polynomial()} == {"exz", "eyz"}
    assert {term.pw2strng() for term in annihilation.return_polynomial()} == {"exz", "eyz"}


def test_pauli_term_import_is_canonical_parent_class() -> None:
    term = PauliTerm(2, ps="ez", pc=1.0)

    assert term.pw2strng() == "ez"
