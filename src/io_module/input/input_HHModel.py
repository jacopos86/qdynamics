from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   Holstein-Hubbard Model Input
#

class HHModelInput(AbstractInput):
    """
    HH Model input class for from YAML input.
    """
    def __init__(self):
        # structural parameters
        self.L = None
        # electronic parameters
        self.t = None
        self.U = None
        self.dv = None
        # vibrational parameters
        self.w0 = None
        self.g = None
        self.n_ph_max = None
        # boundary conditions
        self.boundary = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract HH model parameters"""
        # chain length
        self.L = self._data.get("L")
        # hopping parameter
        self.t = self._data.get("hopping_parameter")
        # Coulomb interaction
        self.U = self._data.get("U")
        # potential offset
        self.dv = self._data.get("dv")
        # vibrational frequency
        self.w0 = self._data.get("w0")
        # electron-phonon coupling
        self.g = self._data.get("g")
        # maximum phonon number
        self.n_ph_max = self._data.get("n_ph_max")
        # boundary conditions
        self.boundary = self._data.get("boundary", "periodic")
    # validation
    def _validate(self):
        """Check that all required fields exist"""
        if not self.t:
            log.error("Missing hopping_parameter in input data")
        if not self.L:
            log.error("Missing L in input data")
        if not self.U:
            log.error("Missing U in input data")
        if self.dv is None:
            log.error("Missing dv in input data")
        if not self.w0:
            log.error("Missing w0 in input data")
        if self.g is None:
            log.error("Missing g in input data")
        if not self.n_ph_max:
            log.error("Missing n_ph_max in input data")