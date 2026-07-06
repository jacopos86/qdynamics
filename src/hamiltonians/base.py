

class BaseHamiltonianBuilder(ABC):
    """
    Base class for all Hamiltonian builders.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def build(self):
        """
        Return ProblemHamiltonianData
        """
        pass


def validate_required(params, required_keys):
    for k in required_keys:
        if k not in params:
            raise ValueError(f"Missing required parameter: {k}")