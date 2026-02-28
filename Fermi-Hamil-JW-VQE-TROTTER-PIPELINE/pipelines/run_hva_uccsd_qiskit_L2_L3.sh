#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/run_hva_uccsd_qiskit_L2_L3.sh [--heavy] [--artifacts-dir <path>] [--python-bin <path>] [-- ...extra compare flags]

Profiles:
  default (smoke): fast sanity run for L=2,3
  --heavy: convergence-oriented run with larger VQE/Trotter settings
USAGE
}

PROFILE="smoke"
ARTIFACTS_DIR="artifacts/hva_uccsd_runs"
PYTHON_BIN_DEFAULT="/opt/anaconda3/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --heavy)
      PROFILE="heavy"
      shift
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: python executable not found: ${PYTHON_BIN}" >&2
    exit 2
  fi
fi

# Shared physical defaults used in current HVA/UCCSD comparisons.
T_VAL="1.0"
U_VAL="4.0"
DV_VAL="0.0"
BOUNDARY="periodic"
ORDERING="blocked"
T_FINAL="20.0"
SUZUKI_ORDER="2"
L_VALUES="2,3"

if [[ "${PROFILE}" == "heavy" ]]; then
  NUM_TIMES="401"
  TROTTER_STEPS="128"
  HC_REPS="6"
  HC_RESTARTS="6"
  HC_MAXITER="2200"
  QK_REPS="6"
  QK_RESTARTS="6"
  QK_MAXITER="2200"
  COMPARE_PDF_FLAGS=(--with-per-l-pdfs)
else
  NUM_TIMES="81"
  TROTTER_STEPS="32"
  HC_REPS="2"
  HC_RESTARTS="1"
  HC_MAXITER="200"
  QK_REPS="2"
  QK_RESTARTS="1"
  QK_MAXITER="200"
  COMPARE_PDF_FLAGS=(--skip-pdf)
fi

echo "Running HVA/UCCSD vs Qiskit compare for L=${L_VALUES} (profile=${PROFILE})"

CMD=(
  "${PYTHON_BIN}" -m pipelines.compare_hardcoded_vs_qiskit_pipeline
  --l-values "${L_VALUES}"
  --t "${T_VAL}" --u "${U_VAL}" --dv "${DV_VAL}"
  --boundary "${BOUNDARY}" --ordering "${ORDERING}"
  --t-final "${T_FINAL}" --num-times "${NUM_TIMES}"
  --suzuki-order "${SUZUKI_ORDER}" --trotter-steps "${TROTTER_STEPS}"
  --hardcoded-vqe-ansatzes "uccsd,hva"
  --hardcoded-vqe-reps "${HC_REPS}" --hardcoded-vqe-restarts "${HC_RESTARTS}" --hardcoded-vqe-seed 7 --hardcoded-vqe-maxiter "${HC_MAXITER}"
  --qiskit-vqe-reps "${QK_REPS}" --qiskit-vqe-restarts "${QK_RESTARTS}" --qiskit-vqe-seed 7 --qiskit-vqe-maxiter "${QK_MAXITER}"
  --initial-state-source vqe
  --skip-qpe
  "${COMPARE_PDF_FLAGS[@]}"
  --artifacts-dir "${ARTIFACTS_DIR}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "Done. Artifacts written under: ${ARTIFACTS_DIR}"
