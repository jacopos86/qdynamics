from src.utilities.log import log
from src.quantum.pauli_letters_module import PauliLetter
from src.set_param_object import p

#
#   This module defines the basic Pauli term entity
#   pt = c_i |zyxz...xxz>
#   each term is a dictionary with key -> string of |xyxz...>
#   pointing to a complex coefficient
#   we override basic operations like +, +=, -, -=

class PauliTerm:
    def __init__(self, nq, ps=None, pc=None, pl_seq=None):
        # pw is a list of pauli letters -> word
        self.p_coeff = None
        self.pw = None
        self.__nq = nq
        if ps is not None:
            assert len(ps) == self.__nq
        if ps is not None and pc is not None:
            self.__set_pauli_term(ps, pc)
        elif pl_seq is not None and pc is not None:
            self.__set_pauli_term_from_letters(pl_seq, pc)
        else:
            log.error("WRONG INITIALIZATION")
    def __set_pauli_term(self, ps, pc):
        pl_seq = []
        for iq in range(self.__nq):
            pl = PauliLetter(symbol=ps[iq])
            pl_seq.append(pl)
        self.pw = pl_seq
        self.p_coeff = pc
    def __set_pauli_term_from_letters(self, pl_seq, p_coeff):
        self.p_coeff = p_coeff
        self.pw = []
        for pl in pl_seq:
            self.p_coeff *= pl.phase
            pl.phase = 1.
            self.pw.append(pl)
    def nqubit(self):
        return self.__nq
    def __mul__(self, pt):
        if not isinstance(pt, PauliTerm):
            return NotImplemented
        pl_seq = []
        for iq in range(self.__nq):
            r = self.pw[iq]*pt.pw[iq]
            pl_seq.append(r)
        return PauliTerm(self.__nq, pc=self.p_coeff*pt.p_coeff, pl_seq=pl_seq)
    def pw2strng(self):
        strng = ""
        for iq in range(self.__nq):
            strng += self.pw[iq].symbol
        return strng
    def pw2sparsePauliOp(self):
        strng = ""
        for iq in range(self.__nq):
            strng += self.pw[iq].symbol.upper().replace("E", "I")
        return (strng, self.p_coeff)
    def visualize(self):
        strng = "\t " + str(self.p_coeff) + " "
        for iq in range(self.__nq):
            strng += self.pw[iq].symbol
        log.info("\t " + p.sep)
        log.info(strng)
        log.info("\t " + p.sep)