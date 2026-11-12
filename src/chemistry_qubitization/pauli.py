"""Self-contained Pauli algebra used by the chemistry qubitizer.

Words are stored in display/matrix order, ``q_(n-1) ... q_0``, using ``e``
for the identity.  The small compatibility surface intentionally matches the
API consumed by this package and its downstream JSON users.
"""
from __future__ import annotations

from numbers import Number


_PRODUCTS = {
    ("e", "e"): (1, "e"), ("e", "x"): (1, "x"),
    ("e", "y"): (1, "y"), ("e", "z"): (1, "z"),
    ("x", "e"): (1, "x"), ("y", "e"): (1, "y"),
    ("z", "e"): (1, "z"), ("x", "x"): (1, "e"),
    ("y", "y"): (1, "e"), ("z", "z"): (1, "e"),
    ("x", "y"): (1j, "z"), ("y", "x"): (-1j, "z"),
    ("y", "z"): (1j, "x"), ("z", "y"): (-1j, "x"),
    ("z", "x"): (1j, "y"), ("x", "z"): (-1j, "y"),
}


class PauliTerm:
    """One coefficient-weighted tensor product of Pauli matrices."""

    def __init__(self, nq, ps=None, pc=None):
        if type(nq) is not int or nq < 1:
            raise ValueError("nq must be a positive integer")
        if ps is None or pc is None or len(ps) != nq or set(ps) - set("exyz"):
            raise ValueError("ps must be an e/x/y/z word of length nq")
        self._nq = nq
        self._word = ps
        self.p_coeff = pc

    def nqubit(self):
        return self._nq

    def pw2strng(self):
        return self._word

    def pw2sparsePauliOp(self):
        return self._word.upper().replace("E", "I"), self.p_coeff

    def __mul__(self, other):
        if not isinstance(other, PauliTerm):
            return NotImplemented
        if self._nq != other._nq:
            raise ValueError("Cannot multiply Pauli terms with different qubit counts")
        coefficient = self.p_coeff * other.p_coeff
        word = []
        for left, right in zip(self._word, other._word):
            phase, symbol = _PRODUCTS[left, right]
            coefficient *= phase
            word.append(symbol)
        return PauliTerm(self._nq, ps="".join(word), pc=coefficient)


class PauliPolynomial:
    """A reduced sum of :class:`PauliTerm` objects."""

    def __init__(self, repr_mode="JW", pol=None):
        if repr_mode != "JW":
            raise ValueError("Only the JW representation is supported")
        self._repr_mode = repr_mode
        self._pol = list(pol or ())
        self._reduce()

    def get_nq(self):
        if not self._pol:
            raise ValueError("An empty PauliPolynomial has no qubit count")
        return self._pol[0].nqubit()

    def return_polynomial(self):
        return self._pol

    def count_number_terms(self):
        return len(self._pol)

    def add_term(self, term):
        if not isinstance(term, PauliTerm):
            raise TypeError("term must be a PauliTerm")
        self._pol.append(term)

    def _check_compatible(self, other):
        if self._repr_mode != other._repr_mode:
            raise ValueError("Pauli representations do not match")
        if self._pol and other._pol and self.get_nq() != other.get_nq():
            raise ValueError("Pauli polynomials have different qubit counts")

    def __add__(self, other):
        if isinstance(other, PauliPolynomial):
            self._check_compatible(other)
            return PauliPolynomial(self._repr_mode, [*self._pol, *other._pol])
        if isinstance(other, Number):
            return self + PauliPolynomial(
                self._repr_mode,
                [PauliTerm(self.get_nq(), ps="e" * self.get_nq(), pc=other)],
            )
        return NotImplemented

    def __iadd__(self, other):
        result = self + other
        if result is NotImplemented:
            return NotImplemented
        self._pol = result._pol
        return self

    def __sub__(self, other):
        if isinstance(other, (PauliPolynomial, Number)):
            return self + (-1) * other
        return NotImplemented

    def __isub__(self, other):
        result = self - other
        if result is NotImplemented:
            return NotImplemented
        self._pol = result._pol
        return self

    def __mul__(self, other):
        if isinstance(other, PauliPolynomial):
            self._check_compatible(other)
            return PauliPolynomial(
                self._repr_mode,
                [left * right for left in self._pol for right in other._pol],
            )
        if isinstance(other, Number):
            return PauliPolynomial(
                self._repr_mode,
                [PauliTerm(term.nqubit(), ps=term.pw2strng(), pc=term.p_coeff * other)
                 for term in self._pol],
            )
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exponent):
        if type(exponent) is not int or exponent < 0:
            raise ValueError("Only non-negative integer powers are supported")
        if exponent == 0:
            nq = self.get_nq()
            return PauliPolynomial(self._repr_mode, [PauliTerm(nq, ps="e" * nq, pc=1)])
        result = PauliPolynomial(self._repr_mode, self._pol)
        for _ in range(exponent - 1):
            result *= self
        return result

    def _reduce(self):
        coefficients = {}
        qubit_counts = {}
        for term in self._pol:
            word = term.pw2strng()
            coefficients[word] = coefficients.get(word, 0j) + complex(term.p_coeff)
            qubit_counts[word] = term.nqubit()
        self._pol = [
            PauliTerm(qubit_counts[word], ps=word, pc=coefficient)
            for word, coefficient in coefficients.items() if coefficient != 0
        ]


def _fermion_operator(n_qubits, index, y_coefficient):
    if type(index) is not int or not 0 <= index < n_qubits:
        raise ValueError("index must satisfy 0 <= index < n_qubits")
    suffix = "z" * index
    prefix = "e" * (n_qubits - index - 1)
    return PauliPolynomial("JW", [
        PauliTerm(n_qubits, ps=prefix + "x" + suffix, pc=0.5),
        PauliTerm(n_qubits, ps=prefix + "y" + suffix, pc=y_coefficient),
    ])


def fermion_plus_operator(repr_mode, n_qubits, index):
    if repr_mode != "JW":
        raise ValueError("Only the JW representation is supported")
    return _fermion_operator(n_qubits, index, -0.5j)


def fermion_minus_operator(repr_mode, n_qubits, index):
    if repr_mode != "JW":
        raise ValueError("Only the JW representation is supported")
    return _fermion_operator(n_qubits, index, 0.5j)
