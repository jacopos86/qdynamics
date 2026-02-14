import numpy as np

from ._standalone_pauli_words import PauliTerm

try:
    from pydephasing.set_param_object import p
except Exception:  # pragma: no cover - local fallback when global params are absent
    class _FallbackParams:
        sep = "-" * 40

    p = _FallbackParams()

try:
    from pydephasing.utilities.log import log
except Exception:  # pragma: no cover - local fallback when utilities package is absent
    class _FallbackLog:
        @staticmethod
        def error(msg):
            raise ValueError(msg)

        @staticmethod
        def info(msg):
            print(msg)

    log = _FallbackLog()


class PauliPolynomial:
    def __init__(self, repr_mode, pol=None):
        self._repr_mode = repr_mode
        self._pol = []
        if pol is not None:
            self._pol.extend(pol)
            self._reduce()

    def return_polynomial(self):
        return self._pol

    def count_number_terms(self):
        return len(self._pol)

    def add_term(self, pt):
        self._pol.append(pt)

    def __add__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        assert self._repr_mode == pp._repr_mode
        new_pol = self._pol + pp.return_polynomial()
        return PauliPolynomial(self._repr_mode, new_pol)

    def __iadd__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        assert self._repr_mode == pp._repr_mode
        self._pol.extend(pp.return_polynomial())
        self._reduce()
        return self

    def __mul__(self, pp):
        pp_new = PauliPolynomial(self._repr_mode)
        n1 = self.count_number_terms()
        if isinstance(pp, PauliPolynomial):
            assert self._repr_mode == pp._repr_mode
            n2 = pp.count_number_terms()
            for i1 in range(n1):
                pt_1 = self._pol[i1]
                for i2 in range(n2):
                    pt_2 = pp.return_polynomial()[i2]
                    pt_new = pt_1 * pt_2
                    pp_new.add_term(pt_new)
        elif isinstance(pp, (int, float, complex)):
            for i1 in range(n1):
                pt = PauliTerm(
                    self._pol[i1].nqubit(),
                    pc=self._pol[i1].p_coeff * pp,
                    pl_seq=self._pol[i1].pw,
                )
                pp_new.add_term(pt)
        else:
            return NotImplemented
        pp_new._reduce()
        return pp_new

    def __rmul__(self, other):
        if not isinstance(other, (int, float, complex)):
            return NotImplemented
        pp_new = PauliPolynomial(self._repr_mode)
        n1 = self.count_number_terms()
        for i1 in range(n1):
            pt = PauliTerm(
                self._pol[i1].nqubit(),
                pc=self._pol[i1].p_coeff * other,
                pl_seq=self._pol[i1].pw,
            )
            pp_new.add_term(pt)
        pp_new._reduce()
        return pp_new

    def _reduce(self):
        pol_temp = list(np.copy(self._pol))
        self._pol = []
        while len(pol_temp) > 0:
            pt = pol_temp.pop()
            strng = pt.pw2strng()
            equal = False
            for i in range(len(self._pol)):
                if self._pol[i].pw2strng() == strng:
                    equal = True
                    self._pol[i].p_coeff += pt.p_coeff
                    if abs(self._pol[i].p_coeff) < 1.0e-7:
                        del self._pol[i]
                    break
            if not equal:
                self._pol.append(pt)

    def visualize_polynomial(self):
        n = self.count_number_terms()
        log.info("\t " + p.sep)
        for i in range(n):
            pt = self._pol[i]
            strng = "\t " + str(i) + " -> " + str(pt.p_coeff) + " "
            for iq in range(len(pt.pw)):
                strng += pt.pw[iq].symbol
            log.info(strng)
        log.info("\t " + p.sep)


class fermion_plus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)

    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j + 1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j + 1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=-0.5j))
        self._reduce()


class fermion_minus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)

    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j + 1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j + 1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5j))
        self._reduce()
