"""Compatibility wrapper for ``pipelines.time_dynamics.runners.generic_from_adapt_artifact``."""
from pipelines.time_dynamics.runners import generic_from_adapt_artifact as _canonical_module
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import *  # noqa: F401,F403
from pipelines.time_dynamics.runners.generic_from_adapt_artifact import main as main  # noqa: F401

try:
    __all__ = _canonical_module.__all__
except AttributeError:
    __all__ = [name for name in dir(_canonical_module) if not name.startswith("_")]

del _canonical_module

if __name__ == "__main__":
    raise SystemExit(main())
