from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#    electronic system input MODEL
#

class ElecSysInput(AbstractInput):
    """
    Electronic system input class for from YAML input.
    """
    def __init__(self):
        # charge
        self.charge = None
        # multiplicity
        self.multiplicity = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract variables"""
        self.charge = self._data.get("charge")
        self.multiplicity = self._data.get("multiplicity")
    # validation
    def _validate(self):
        """Check that all required fields exist and files are accessible"""
        if self.charge is None:
            log.error("charge must be given in input")
        if self.multiplicity is None:
            log.error("multiplicity must be given")