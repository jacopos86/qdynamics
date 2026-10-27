"""Molecular one- and two-electron tensors in blocked-spin JW ordering."""
from .pauli import (
    PauliPolynomial, fermion_plus_operator, fermion_minus_operator,
)
from .pauli import PauliTerm


def one_body(h1, n_qubits):
    n = len(h1)
    result = PauliPolynomial("JW")
    for spin in (0, 1):
        for p in range(n):
            for q in range(n):
                if h1[p, q] != 0:
                    result += float(h1[p, q]) * (fermion_plus_operator("JW", n_qubits, p + spin*n) * fermion_minus_operator("JW", n_qubits, q + spin*n))
    return result


def electronic(integrals, n_qubits):
    n = len(integrals.h1)
    result = PauliPolynomial("JW", [PauliTerm(n_qubits, ps="e" * n_qubits, pc=integrals.nuclear_repulsion)])
    result += one_body(integrals.h1, n_qubits)
    creators = [fermion_plus_operator("JW", n_qubits, p) for p in range(2*n)]
    annihilators = [fermion_minus_operator("JW", n_qubits, p) for p in range(2*n)]
    for p in range(2*n):
        for q in range(2*n):
            for r in range(p//n*n, (p//n+1)*n):
                for s in range(q//n*n, (q//n+1)*n):
                    coefficient = 0.5 * float(integrals.h2[p % n, r % n, q % n, s % n])
                    if coefficient != 0:
                        result += coefficient * (creators[p] * creators[q] * annihilators[s] * annihilators[r])
    return result
