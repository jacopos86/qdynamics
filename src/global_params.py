from pathlib import Path
import os
import src

#
MPI_ROOT = 0
ngpus = 0
# === Repo root MUST be set via environment variable ===

# === Get path from module ===
PACKAGE_DIR = Path(src.__path__[0]).resolve()
