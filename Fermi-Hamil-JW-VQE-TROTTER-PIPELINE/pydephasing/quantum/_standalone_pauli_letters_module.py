try:
    from pydephasing.utilities.log import log
except Exception:  # pragma: no cover - local fallback when utilities package is absent
    class _FallbackLog:
        @staticmethod
        def error(msg):
            raise ValueError(msg)

    log = _FallbackLog()


symbol_product_map = {
    ("e", "e"): (1.0, "e"),
    ("e", "x"): (1.0, "x"),
    ("e", "y"): (1.0, "y"),
    ("e", "z"): (1.0, "z"),
    ("x", "e"): (1.0, "x"),
    ("x", "x"): (1.0, "e"),
    ("x", "y"): (1j, "z"),
    ("x", "z"): (-1j, "y"),
    ("y", "e"): (1.0, "y"),
    ("y", "x"): (-1j, "z"),
    ("y", "y"): (1.0, "e"),
    ("y", "z"): (1j, "x"),
    ("z", "e"): (1.0, "z"),
    ("z", "x"): (1j, "y"),
    ("z", "y"): (-1j, "x"),
    ("z", "z"): (1.0, "e"),
}


class PauliLetter:
    def __init__(self, symbol, phase=None):
        if not isinstance(symbol, str) or len(symbol) != 1:
            log.error("Symbol must be a single character string.")
        self.symbol = symbol
        if phase is None:
            self.phase = 1.0
        else:
            self.phase = phase

    def __mul__(self, pl):
        if not isinstance(pl, PauliLetter):
            log.error("product must be with pauli letter")
        phase, sym = symbol_product_map[(self.symbol, pl.symbol)]
        return PauliLetter(symbol=sym, phase=phase)
