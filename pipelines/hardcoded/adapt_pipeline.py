#!/usr/bin/env python3
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_module = import_module("pipelines.static_adapt.adapt_pipeline")


class _ForwardingModule(ModuleType):
    _LOCAL_ATTRS = {
        "__class__",
        "__dict__",
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
        "_LOCAL_ATTRS",
    }

    def __getattribute__(self, name):
        if name in _ForwardingModule._LOCAL_ATTRS:
            return super().__getattribute__(name)
        try:
            return getattr(_module, name)
        except AttributeError:
            return super().__getattribute__(name)

    def __getattr__(self, name):
        return getattr(_module, name)

    def __setattr__(self, name, value):
        setattr(_module, name, value)
        return super().__setattr__(name, value)


_wrapper_module = sys.modules.get(__name__)
if _wrapper_module is not None and _wrapper_module is not _module:
    _wrapper_module.__class__ = _ForwardingModule

for _name, _value in _module.__dict__.items():
    if _name in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__", "__builtins__"}:
        continue
    globals()[_name] = _value

sys.modules[__name__] = _module

if __name__ == "__main__":
    _main = getattr(_module, "main", None)
    if callable(_main):
        _result = _main()
        if isinstance(_result, int):
            raise SystemExit(_result)
