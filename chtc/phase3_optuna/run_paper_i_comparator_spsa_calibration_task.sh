#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${2:-${PAPER_I_COMPARATOR_SPSA_CALIBRATION_RECORDS_PATH:-chtc/phase3_optuna/input/paper_i_comparator_spsa_calibration_v1_smoke/paper_i_comparator_spsa_calibration_records.tsv}}"
OUT_ROOT="${3:-raw_outputs/paper_i_comparator_spsa_calibration_v1/records/${RECORD_ID}}"
CONFIG_PATH="${4:-${PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_PATH:-chtc/phase3_optuna/config/paper_i_comparator_spsa_calibration_v1_smoke.json}}"

mkdir -p logs "$(dirname "$OUT_ROOT")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

python - <<'PY'
import importlib.util
import sys

checks = {
    "optuna": "optuna",
    "qiskit_algorithms": "qiskit-algorithms",
}
missing = [package for module, package in checks.items() if importlib.util.find_spec(module) is None]
if missing:
    sys.stderr.write(
        "Missing required Python dependencies for Paper-I comparator SPSA calibration: "
        + ", ".join(missing)
        + ". Install the CHTC requirements/image before launching.\n"
    )
    raise SystemExit(127)
try:
    from qiskit_algorithms.optimizers import SPSA  # noqa: F401
except Exception as exc:
    sys.stderr.write(f"Qiskit Algorithms is present but SPSA is unavailable: {exc}\n")
    raise SystemExit(127)
PY

python -u -m chtc.phase3_optuna.paper_i_comparator_spsa_calibration_runner \
  "$RECORD_ID" "$RECORDS_PATH" "$OUT_ROOT" --config "$CONFIG_PATH"
