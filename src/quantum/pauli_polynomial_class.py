import numpy as np
from src.set_param_object import p
from src.utilities.log import log
from src.quantum.qubitization_module import PauliTerm

#
#   Pauli polynomial class definition
#

class PauliPolynomial:
    def __init__(self, repr_mode, pol=None):
        self._repr_mode = repr_mode
        self._pol = []
        if pol is not None:
            self._pol.extend(pol)
            self._reduce()
    def get_nq(self):
        if len(self._pol) > 0:
            nq = self._pol[0].nqubit()
        else:
            log.error("Cannot add a constant to empty PauliPolynomial without qubit count")
        return nq
    def return_polynomial(self):
        return self._pol
    def count_number_terms(self):
        return len(self._pol)
    def add_term(self, pt):
        self._pol.append(pt)
    def __add__(self, pp):
        if isinstance(pp, PauliPolynomial):
            assert self._repr_mode == pp._repr_mode
            new_pol = self._pol + pp.return_polynomial()
        elif isinstance(pp, (float, complex)):
            nq = self.get_nq()
            const_term = PauliTerm(nq, pc=pp, ps="e"*nq)
            new_pol = self._pol + [const_term]
        else:
            return NotImplemented
        return PauliPolynomial(self._repr_mode, new_pol)
    def __iadd__(self, pp):
        if isinstance(pp, PauliPolynomial):
            assert self._repr_mode == pp._repr_mode
            self._pol.extend(pp.return_polynomial())
        elif isinstance(pp, (float, complex)):
            nq = self.get_nq()
            const_term = PauliTerm(nq, pc=pp, ps="e"*nq)
            self.add_term(const_term)
        else:
            return NotImplemented
        self._reduce()
        return self
    def __sub__(self, pp):
        if isinstance(pp, PauliPolynomial):
            assert self._repr_mode == pp._repr_mode
            min_pp = (-1.0) * pp
            new_pol = self._pol + min_pp
        elif isinstance(pp, (float, complex)):
            nq = self.get_nq()
            const_term = PauliTerm(nq, pc=-pp, ps="e"*nq)
            new_pol = self._pol + [const_term]
        else:
            return NotImplemented
        return PauliPolynomial(self._repr_mode, new_pol)
    def __isub__(self, pp):
        if isinstance(pp, PauliPolynomial):
            assert self._repr_mode == pp._repr_mode
            min_pp = (-1.0) * pp
            self._pol.extend(min_pp.return_polynomial())
        elif isinstance(pp, (float, complex)):
            nq = self.get_nq()
            const_term = PauliTerm(nq, pc=-pp, ps="e"*nq)
            self.add_term(const_term)
        else:
            return NotImplemented
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
                pt = PauliTerm(self._pol[i1].nqubit(), pc=self._pol[i1].p_coeff*pp, pl_seq=self._pol[i1].pw)
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
            pt = PauliTerm(self._pol[i1].nqubit(), pc=self._pol[i1].p_coeff*other, pl_seq=self._pol[i1].pw)
            pp_new.add_term(pt)
        pp_new._reduce()
        return pp_new
    def __pow__(self, exponent):
        if isinstance(exponent, int) and exponent >= 0:
            if exponent == 0:
                nq = self.get_nq()
                Id = PauliPolynomial(self._repr_mode)
                Id.add_term(PauliTerm(nq, pc=1.0, ps="e"*nq))
                return Id
            result = PauliPolynomial(self._repr_mode, self._pol.copy())
            for _ in range(exponent-1):
                result = result * self
            return result
        else:
            log.error("Only non-negative integer powers are supported for PauliPolynomial")
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
                    if abs(self._pol[i].p_coeff) < 1.e-7:
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

#
#   Define cj^+ fermionic operator
#   in qubit representation

class fermion_plus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)
    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=-0.5j))
        self._reduce()

#
#   Define cj fermionic operator
#   in qubit representation

class fermion_minus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)
    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5j))
        self._reduce()