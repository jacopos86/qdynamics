from abc import ABC, abstractmethod
import yaml
from src.common.units import Q_
from src.utilities.log import log

#
#   pure abstract input class
#

class AbstractInput(ABC):
    _data = None
    """Pure abstract base for all input classes."""
    # load yaml file
    def _load_yaml(self, yaml_file: str) -> dict:
        try:
            with open(yaml_file, "r") as f:
                return yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find YAML file: {yaml_file}")
    # read yaml
    def read_yml_data(self, yaml_file: str):
        """Read YAML file and populate BaseInput fields."""
        # load YAML
        self._data = self._load_yaml(yaml_file)
        # parse common data
        self._parse_data()
        # validate directories
        self._validate()
    # parsing method
    @abstractmethod
    def _parse_data(self):
        """Read input data """
        raise NotImplementedError
    # validation method
    @abstractmethod
    def _validate(self):
        raise NotImplementedError
    @staticmethod
    def parse_quantity_from_dict(input_dict: dict, required_keys=("units", "value"), desc="quantity") -> Q_:
        """
        Extract a Quantity from a dictionary, checking required keys.
        Works for both scalar and array values.
        Args:
            input_dict: dict containing quantity info (keys: units, value or values)
            required_keys: keys to check in the dict
            desc: description for error messages
        Returns:
            Q_ instance with the proper unit
        """
        if input_dict is None:
            return None
        # check required keys
        missing = [k for k in required_keys if k not in input_dict]
        if missing:
            log.error(f"{desc} dictionary missing keys: {missing}")
        # extract value(s)
        val = input_dict.get(required_keys[1])
        units = input_dict.get(required_keys[0])
        if units is None or val is None:
            log.error(f"{desc} dictionary keys cannot be None")
        # convert to Quantity
        try:
            q = Q_(val, units)  # Q_ handles scalar or array
        except Exception as e:
            log.error(f"Failed to convert {desc} to Quantity: {e}")
        return q