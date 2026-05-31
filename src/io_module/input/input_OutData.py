from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   Output Data Input
#

class OutDataInput(AbstractInput):
    """
    Output data input class for specifying output files and formats from YAML input.
    """
    _ALLOWED_FORMATS = {"json", "yaml", "hdf5", "pickle"}
    def __init__(self):
        # output files
        self.output_json = None
        self.output_pdf = None
        self.output_hdf5 = None
        # control parameters
        self.skip_pdf = False
        self.output_format = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract output parameters"""
        # JSON output file path
        self.output_json = self._data.get("output_json")
        # PDF summary file path
        self.output_pdf = self._data.get("output_pdf")
        # HDF5 data file path
        self.output_hdf5 = self._data.get("output_hdf5")
        # Skip PDF generation
        self.skip_pdf = self._data.get("skip_pdf", False)
        # Output format preference
        self.output_format = self._data.get("output_format", "json")
    # validation
    def _validate(self):
        """Validate output parameters"""
        if self.output_format not in self._ALLOWED_FORMATS:
            log.error(f"{self.output_format!r} not allowed for output_format. Must be one of {self._ALLOWED_FORMATS}")