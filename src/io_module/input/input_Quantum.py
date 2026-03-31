from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   Quantum model class
#

class QuantumModelInput(AbstractInput):
    """
    QUANTUM input class for from YAML input.
    """
    _ALLOWED_F2Q_MAPS = {"JW"}
    def __init__(self):
        self.init_mode = None
        # qubitization mode
        self.F2Q = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract quantum variables"""
        # F2Q
        self.F2Q = self._data.get("fermion2qubit")
    # validation
    def _validate(self):
        if self.F2Q not in self._ALLOWED_F2Q_MAPS:
            log.error(f"{self.F2Q!r} not allowed. Must be one of {self._ALLOWED_F2Q_MAPS}")