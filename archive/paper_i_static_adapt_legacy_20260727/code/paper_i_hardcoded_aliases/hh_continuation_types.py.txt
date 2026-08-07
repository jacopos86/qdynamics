#!/usr/bin/env python3
"""Compatibility alias for scaffold continuation types."""

from importlib import import_module
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_module = import_module("pipelines.scaffold.hh_continuation_types")
for _name, _value in _module.__dict__.items():
    if _name in {
        "__name__",
        "__package__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__builtins__",
    }:
        continue
    globals()[_name] = _value

sys.modules[__name__] = _module
