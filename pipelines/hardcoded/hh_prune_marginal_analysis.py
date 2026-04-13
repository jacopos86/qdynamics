#!/usr/bin/env python3
from importlib import import_module
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_module = import_module("pipelines.static_adapt.hh_prune_marginal_analysis")
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
