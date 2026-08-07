#!/usr/bin/env python3
import importlib.util
from pathlib import Path
SCRIPT = Path(__file__).resolve().parents[2] / "build_paper_i_hh_macro_only_fidelity_successor_20260719.py"
spec = importlib.util.spec_from_file_location("macro_fidelity_successor_builder", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
if __name__ == "__main__":
    module.verify(Path(__file__).resolve().parent, run_archive_preflight=False)
    print("macro fidelity remaining2 successor verification passed")
