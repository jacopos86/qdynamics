from pathlib import Path
from src.io_module.input.core import AbstractInput
from src.utilities.log import log

class InputError(Exception):
    """Raised when input data is invalid."""
    pass

class BaseInput(AbstractInput):
    """
    Base input class for parsing core directories from YAML input.
    Attributes:
        work_dir: working directory
        write_dir: output directory
        sep: separator string for logging
    """
    _ALLOWED_INIT_MODES = {"thermal", "input", "optimization"}
    def __init__(self):
        # working directory
        self.work_dir = None
        # write directory
        self.write_dir = None
        # separator
        self.sep = "*"*94
    def _parse_data(self):
        """Read data and extract work_dir, write_dir, and sep."""
        # extract work_dir
        self.work_dir = self._data.get("working_dir")
        # extract write_dir and create it if needed
        self.write_dir = self._data.get("output_dir")
    def _ensure_write_dir(self):
        """Create the write directory if it does not exist."""
        Path(self.write_dir).mkdir(parents=True, exist_ok=True)
    def _validate(self):
        """Ensure that work_dir and write_dir are defined and create write_dir."""
        if not self.work_dir:
            raise InputError("Missing 'working_dir' in input YAML.")
        if not self.write_dir:
            raise InputError("Missing 'output_dir' in input YAML.")
        self._ensure_write_dir()