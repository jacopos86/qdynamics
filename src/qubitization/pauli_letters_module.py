from src.utilities.log import log

#
#   Define Pauli letter class
#

symbol_product_map = {
    ('e', 'e'): (1., 'e'),
    ('e', 'x'): (1., 'x'),
    ('e', 'y'): (1., 'y'),
    ('e', 'z'): (1., 'z'),
    ('x', 'e'): (1., 'x'),
    ('x', 'x'): (1., 'e'),
    ('x', 'y'): (1j, 'z'),
    ('x', 'z'): (-1j, 'y'),
    ('y', 'e'): (1., 'y'),
    ('y', 'x'): (-1j, 'z'),
    ('y', 'y'): (1., 'e'),
    ('y', 'z'): (1j, 'x'),
    ('z', 'e'): (1., 'z'),
    ('z', 'x'): (1j, 'y'),
    ('z', 'y'): (-1j, 'x'),
    ('z', 'z'): (1., 'e')
}

class PauliLetter:
    def __init__(self, symbol, phase=None):
        if not isinstance(symbol, str) or len(symbol) != 1:
            log.error("Symbol must be a single character string.")
        self.symbol = symbol
        if phase is None:
            self.phase = 1.
        else:
            self.phase = phase
    def __mul__(self, pl):
        if not isinstance(pl, PauliLetter):
            log.error("product must be with pauli letter")
        phase, sym = symbol_product_map[(self.symbol, pl.symbol)]
        return PauliLetter(symbol=sym, phase=phase)