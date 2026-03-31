from src.utilities.log import log

#
#    general Input Combiner class
#

class InputCombiner:
    """
    Dynamically combine multiple input classes into a single object.
    Each class receives the same yaml_file argument during initialization.
    """
    def __init__(self, *input_classes):
        self._initialized = False
        self._inputs = [cls() for cls in input_classes]
    def read_yml_data(self, yaml_file: str):
        if self._initialized:
            log.error("InputCombiner already initialized with a YAML file")
        seen = set()
        for obj in self._inputs:
            obj.read_yml_data(yaml_file)
            for k, v in obj.__dict__.items():
                if k.startswith("_"):
                    continue
                if k in seen:
                    log.error(
                        f"Duplicate attribute '{k}' found in multiple input classes"
                    )
                setattr(self, k, v)
                seen.add(k)
        self._initialized = True
    def __getattr__(self, attr):
        """Fallback to inputs if attribute not found in self."""
        for inp in self._inputs:
            if hasattr(inp, attr):
                return getattr(inp, attr)
        raise AttributeError(f"{attr} not found in any input")