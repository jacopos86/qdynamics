#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

TAG="${TAG:-overnight}"
MODE="${MODE:-smoke_then_full}"
NUM_SEEDS="${NUM_SEEDS:-5}"
SEED_START="${SEED_START:-101}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
BACKGROUND="${BACKGROUND:-1}"
VQE_CAP_S="${VQE_CAP_S:-1200}"
ADAPT_CAP_S="${ADAPT_CAP_S:-1800}"

CMD=(
  "${PYTHON_BIN}" pipelines/exact_bench/overnight_l3_hh_four_method_benchmark.py
  --mode "${MODE}"
  --tag "${TAG}"
  --num-seeds "${NUM_SEEDS}"
  --seed-start "${SEED_START}"
  --vqe-cap-s "${VQE_CAP_S}"
  --adapt-cap-s "${ADAPT_CAP_S}"
)

if [[ "${MAX_ATTEMPTS}" != "0" ]]; then
  CMD+=(--max-attempts "${MAX_ATTEMPTS}")
fi

mkdir -p artifacts/overnight_l3_hh_4method/logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="artifacts/overnight_l3_hh_4method/logs/run_${TAG}_${TS}.log"

echo "Command: ${CMD[*]}"
echo "Log: ${LOG_PATH}"

if [[ "${BACKGROUND}" == "1" ]]; then
  nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
  PID=$!
  echo "Started background benchmark with PID=${PID}"
  echo "Tail with: tail -f ${LOG_PATH}"
else
  "${CMD[@]}" | tee "${LOG_PATH}"
fi

