from pathlib import Path
from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   PSI4 input class
#

class Psi4Input(AbstractInput):
    """
    PSI4 input class for from YAML input.
    """
    # ways to build fock from TEI
    _SCF_TYPES = {
        "pk",
        "df",
        "direct",
        "out_of_core",
        "cd"
    }
    _REFERENCE_TYPES = {
        "rhf",
        "uhf",
        "rohf",
        "rks",
        "uks"
    }
    def __init__(self):
        # input files
        self.coordinate_file = None
        self.basis_set_file = None
        # output
        self.optimized_coordinate_file = None
        self.optimize_geometry = True
        # basis set
        self.basis_set = None
        # optional output controls
        self.write_vibration = False
        self.write_eph = False
        # calculation parameters
        self.psi4_calc_parameters = {
            "scf_type": None,
            "reference": None,
            "method": None,
            "e_converg": None,
            "d_converg": 1.e-8,
            "max_iter": 1000,
            "guess": "sad",
            "soscf": False,
            "soscf_max_iter": 0
        }
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract PSI4 variables"""
        # xyz file
        self.coordinate_file = self._data.get("coordinate_file")
        # optimized coordinate file
        self.optimized_coordinate_file = self._data.get("optimized_coordinate_file")
        self.optimize_geometry = self._data.get("optimize_geometry", True)
        # basis set file
        self.basis_set_file = self._data.get("basis_set_file")
        # basis set
        self.basis_set = self._data.get("basis_set")
        # optional output controls
        self.write_vibration = self._data.get("write_vibration", False)
        self.write_eph = self._data.get("write_eph", False)
        # calculations parameters
        calc_parameters = self._data.get("psi4_calc_parameters")
        self.psi4_calc_parameters["scf_type"] = calc_parameters.get("scf_type")
        self.psi4_calc_parameters["reference"] = calc_parameters.get("reference")
        self.psi4_calc_parameters["method"] = calc_parameters.get("method")
        self.psi4_calc_parameters["e_converg"] = calc_parameters.get("e_converg")
        if calc_parameters.get("d_converg") is not None:
            self.psi4_calc_parameters["d_converg"] = calc_parameters.get("d_converg")
        if calc_parameters.get("max_iter") is not None:
            self.psi4_calc_parameters["max_iter"] = calc_parameters.get("max_iter")
        if calc_parameters.get("guess") is not None:
            self.psi4_calc_parameters["guess"] = calc_parameters.get("guess")
        if calc_parameters.get("soscf") is not None:
            self.psi4_calc_parameters["soscf"] = calc_parameters.get("soscf")
        if calc_parameters.get("soscf_max_iter") is not None:
            self.psi4_calc_parameters["soscf_max_iter"] = calc_parameters.get("soscf_max_iter")
    # validation
    def _validate(self):
        """Check that all required fields exist and files are accessible"""
        if not self.coordinate_file:
            log.error("Missing 'coordinate_file' in input data")
        if not self.optimized_coordinate_file:
            log.error("Missing 'optimized_coordinate_file' in input data")
        if not self.basis_set_file and not self.basis_set:
            log.error("'basis_set_file' or 'basis_set' must be provided")
        if not self.psi4_calc_parameters:
            log.error("Missing 'psi4_calc_parameters' in input data")
        # optional: check specific keys in psi4_calc_parameters
        required_keys = ["scf_type", "reference", "method", "e_converg"]
        missing = [k for k in required_keys 
           if k not in self.psi4_calc_parameters or self.psi4_calc_parameters[k] is None]
        if missing:
            log.error(f"Missing keys in psi4_calc_parameters: {missing}")
        # check scf_type
        if self.psi4_calc_parameters.get("scf_type") not in self._SCF_TYPES:
            log.error(
                f"Invalid scf_type '{self.psi4_calc_parameters.get('scf_type')}'. "
                f"Valid options: {sorted(self._SCF_TYPES)}"
        )
        # check reference
        if self.psi4_calc_parameters.get("reference") not in self._REFERENCE_TYPES:
            log.error.append(
                f"Invalid reference '{self.psi4_calc_parameters.get('reference')}'. "
                f"Valid options: {sorted(self._REFERENCE_TYPES)}"
            )
