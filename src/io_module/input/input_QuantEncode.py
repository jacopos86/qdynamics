from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   Quantum model class
#

class QuantumEncodingInput(AbstractInput):
    """
    QUANTUM encoding input class from YAML input.
    Handles fermion-to-qubit encoding, boson encoding, and fermion mode ordering.
    """
    _ALLOWED_FERMION_ENCODINGS = {"JW"}  # Jordan-Wigner
    _ALLOWED_BOSON_ENCODINGS = {"binary", "unary", "gray"}
    _ALLOWED_ORDERINGS = {"blocked", "interleaved"}  # fermion mode ordering
    def __init__(self):
        # Fermion-to-qubit mapping
        self.fermion_encoding = None
        # Boson encoding (how to represent phonon Hilbert space on qubits)
        self.boson_encoding = None
        # Fermion mode ordering in the Hamiltonian
        self.ordering = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract quantum encoding variables"""
        # Fermion-to-qubit mapping (e.g., Jordan-Wigner)
        self.fermion_encoding = self._data.get("fermion_encoding", "JW")
        # Boson encoding (how to encode phonon operators on qubits)
        self.boson_encoding = self._data.get("boson_encoding", "binary")
        # Fermion mode ordering in the Hamiltonian
        self.ordering = self._data.get("ordering", "blocked")
    # validation
    def _validate(self):
        """Validate quantum encoding parameters"""
        if self.fermion_encoding not in self._ALLOWED_FERMION_ENCODINGS:
            log.error(f"{self.fermion_encoding!r} not allowed for fermion_encoding. Must be one of {self._ALLOWED_FERMION_ENCODINGS}")
        if self.boson_encoding not in self._ALLOWED_BOSON_ENCODINGS:
            log.error(f"{self.boson_encoding!r} not allowed for boson_encoding. Must be one of {self._ALLOWED_BOSON_ENCODINGS}")
        if self.ordering not in self._ALLOWED_ORDERINGS:
            log.error(f"{self.ordering!r} not allowed for ordering. Must be one of {self._ALLOWED_ORDERINGS}")