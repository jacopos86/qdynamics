#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
if command -v apptainer >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v singularity)"
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 127
fi
mkdir -p chtc/time_dynamics_optuna
"$APPTAINER_BIN" build chtc/time_dynamics_optuna/image.sif chtc/time_dynamics_optuna/image.def
"$APPTAINER_BIN" exec --cleanenv --bind "$ROOT:/work" chtc/time_dynamics_optuna/image.sif \
  bash -lc 'cd /work && python --version && python - <<"__DEP_PY__"
import numpy, scipy, optuna, qiskit, qiskit_aer, matplotlib
print("deps-ok", numpy.__version__, scipy.__version__, optuna.__version__, qiskit.__version__, qiskit_aer.__version__, matplotlib.__version__)
__DEP_PY__'
