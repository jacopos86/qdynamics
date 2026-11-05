"""Truncated oscillator operators in binary or one-hot unary encoding."""
from itertools import product
import numpy as np

from .pauli import PauliPolynomial, PauliTerm


def register_size(cutoff, encoding):
    if type(cutoff) is not int or cutoff < 1:
        raise ValueError("Each boson cutoff must be a positive integer")
    if encoding not in {"binary", "unary"}:
        raise ValueError("Encoding must be binary or unary")
    return cutoff.bit_length() if encoding == "binary" else cutoff + 1


def oscillator(cutoff, encoding, offset, n_qubits):
    """Return (number, b + b^dagger) on the selected boson register."""
    width = register_size(cutoff, encoding)
    matrices = {"e": np.eye(2), "x": np.array([[0, 1], [1, 0]]),
                "y": np.array([[0, -1j], [1j, 0]]), "z": np.diag([1, -1])}

    def term(ops, coefficient):
        word = ["e"] * n_qubits
        for qubit, letter in ops.items():
            word[n_qubits - 1 - offset - qubit] = letter
        return PauliTerm(n_qubits, ps="".join(word), pc=coefficient)

    if encoding == "unary":
        number, displacement = [], []
        for level in range(1, cutoff + 1):
            number.extend([term({}, level / 2), term({level: "z"}, -level / 2)])
            for letter in ("x", "y"):
                displacement.append(term({level-1: letter, level: letter}, np.sqrt(level) / 2))
        return PauliPolynomial("JW", number), PauliPolynomial("JW", displacement)
    dim = 1 << width
    b = np.zeros((dim, dim))
    for level in range(1, cutoff + 1):
        b[level-1, level] = np.sqrt(level)
    operators = (b.T @ b, b + b.T)
    terms = [[], []]
    for letters in product("exyz", repeat=width):
        basis = np.array([[1]], dtype=complex)
        for letter in letters:
            basis = np.kron(basis, matrices[letter])
        for index, operator in enumerate(operators):
            coefficient = np.trace(basis @ operator) / dim
            if coefficient != 0:
                terms[index].append(term({width-1-i: letter for i, letter in enumerate(letters)}, coefficient))
    return tuple(PauliPolynomial("JW", values) for values in terms)
