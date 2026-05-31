from src.io_module.input.core import AbstractInput
from src.utilities.log import log

#
#   VQE (Variational Quantum Eigensolver) Input
#

class QuantumVQEInput(AbstractInput):
    """
    VQE input class for ADAPT-VQE and other VQE settings from YAML input.
    """
    _ALLOWED_OPTIMIZERS = {"COBYLA", "SLSQP", "L-BFGS-B", "POWELL"}
    _ALLOWED_INIT_STATES = {"HF", "random", "zero"}
    def __init__(self):
        # ADAPT-VQE settings
        self.adapt_max_depth = None
        self.adapt_eps_grad = None
        self.adapt_maxiter = None
        self.adapt_inner_optimizer = None
        self.initial_state_source = None
    # parsing data
    def _parse_data(self):
        """Parse YAML file and extract VQE parameters"""
        # ADAPT-VQE maximum ansatz depth
        self.adapt_max_depth = self._data.get("adapt_max_depth", 10)
        # ADAPT-VQE gradient threshold
        self.adapt_eps_grad = self._data.get("adapt_eps_grad", 1e-3)
        # ADAPT-VQE maximum iterations
        self.adapt_maxiter = self._data.get("adapt_maxiter", 100)
        # Inner optimizer for ADAPT-VQE
        self.adapt_inner_optimizer = self._data.get("adapt_inner_optimizer", "COBYLA")
        # Initial state preparation method
        self.initial_state_source = self._data.get("initial_state_source", "HF")
    # validation
    def _validate(self):
        """Validate VQE parameters"""
        if self.adapt_inner_optimizer not in self._ALLOWED_OPTIMIZERS:
            log.error(f"{self.adapt_inner_optimizer!r} not allowed for adapt_inner_optimizer. Must be one of {self._ALLOWED_OPTIMIZERS}")
        if self.initial_state_source not in self._ALLOWED_INIT_STATES:
            log.error(f"{self.initial_state_source!r} not allowed for initial_state_source. Must be one of {self._ALLOWED_INIT_STATES}")