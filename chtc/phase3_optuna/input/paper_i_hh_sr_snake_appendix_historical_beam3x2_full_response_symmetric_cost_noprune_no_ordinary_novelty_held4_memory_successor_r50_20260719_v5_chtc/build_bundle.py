#!/usr/bin/env python3
"""Verify the immutable operational-only beam memory successor."""
import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[2] / "build_paper_i_hh_sr_beam_memory_successor_20260719.py"
spec = importlib.util.spec_from_file_location("beam_memory_builder", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
if __name__ == "__main__":
    module.verify(Path(__file__).resolve().parent)
    print("beam memory successor verification passed")
