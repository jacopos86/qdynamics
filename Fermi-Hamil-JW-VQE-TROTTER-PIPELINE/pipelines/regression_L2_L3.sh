#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="/opt/anaconda3/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python not executable at ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p artifacts

# Shared physics settings.
T_VAL="1.0"
U_VAL="4.0"
DV_VAL="0.0"
BOUNDARY="periodic"
ORDERING="blocked"
SUZUKI_ORDER="2"

# Light L=2 run settings.
L2_TROTTER_STEPS="64"
L2_NUM_TIMES="201"
L2_VQE_REPS="2"
L2_VQE_RESTARTS="1"
L2_VQE_MAXITER="120"

# Heavier L=3 run settings.
L3_TROTTER_STEPS="128"
L3_NUM_TIMES="401"
L3_VQE_REPS="2"
L3_VQE_RESTARTS="3"
L3_VQE_MAXITER="600"

# Common QPE settings (diagnostic only; not pass/fail gate).
QPE_EVAL_QUBITS="5"
QPE_SHOTS="256"
QPE_SEED="11"

echo "Running L=2 hardcoded pipeline..."
"${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
  --L 2 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final 20.0 \
  --num-times "${L2_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${L2_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${L2_VQE_REPS}" \
  --vqe-restarts "${L2_VQE_RESTARTS}" \
  --vqe-maxiter "${L2_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --initial-state-source vqe \
  --output-json artifacts/hardcoded_pipeline_L2_reg.json \
  --output-pdf artifacts/hardcoded_pipeline_L2_reg.pdf

echo "Running L=2 qiskit pipeline..."
"${PYTHON_BIN}" pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 2 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final 20.0 \
  --num-times "${L2_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${L2_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${L2_VQE_REPS}" \
  --vqe-restarts "${L2_VQE_RESTARTS}" \
  --vqe-maxiter "${L2_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --initial-state-source vqe \
  --output-json artifacts/qiskit_pipeline_L2_reg.json \
  --output-pdf artifacts/qiskit_pipeline_L2_reg.pdf

echo "Running L=3 hardcoded pipeline..."
"${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
  --L 3 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final 20.0 \
  --num-times "${L3_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${L3_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${L3_VQE_REPS}" \
  --vqe-restarts "${L3_VQE_RESTARTS}" \
  --vqe-maxiter "${L3_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --initial-state-source vqe \
  --output-json artifacts/hardcoded_pipeline_L3_reg.json \
  --output-pdf artifacts/hardcoded_pipeline_L3_reg.pdf

echo "Running L=3 qiskit pipeline..."
"${PYTHON_BIN}" pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 3 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final 20.0 \
  --num-times "${L3_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${L3_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${L3_VQE_REPS}" \
  --vqe-restarts "${L3_VQE_RESTARTS}" \
  --vqe-maxiter "${L3_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --initial-state-source vqe \
  --output-json artifacts/qiskit_pipeline_L3_reg.json \
  --output-pdf artifacts/qiskit_pipeline_L3_reg.pdf

# Option 1 bridge: compare runner expects canonical unsuffixed names.
cp -f artifacts/hardcoded_pipeline_L2_reg.json artifacts/hardcoded_pipeline_L2.json
cp -f artifacts/qiskit_pipeline_L2_reg.json artifacts/qiskit_pipeline_L2.json
cp -f artifacts/hardcoded_pipeline_L3_reg.json artifacts/hardcoded_pipeline_L3.json
cp -f artifacts/qiskit_pipeline_L3_reg.json artifacts/qiskit_pipeline_L3.json

echo "Running compare pipeline for L=2,3..."
"${PYTHON_BIN}" pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3 \
  --no-run-pipelines \
  --artifacts-dir artifacts \
  --with-per-l-pdfs

overall_pass=1
if [[ ! -f artifacts/hardcoded_vs_qiskit_pipeline_summary.json ]]; then
  overall_pass=0
else
  compare_all_pass="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/hardcoded_vs_qiskit_pipeline_summary.json")
data = json.loads(path.read_text(encoding="utf-8"))
print("1" if bool(data.get("all_pass", False)) else "0")
PY
)"
  if [[ "${compare_all_pass}" != "1" ]]; then
    overall_pass=0
  fi
fi

echo "Running manual compare checks..."
for L in 2 3; do
  if "${PYTHON_BIN}" pipelines/manual_compare_jsons.py \
      --hardcoded "artifacts/hardcoded_pipeline_L${L}.json" \
      --qiskit "artifacts/qiskit_pipeline_L${L}.json" \
      --metrics "artifacts/hardcoded_vs_qiskit_pipeline_L${L}_metrics.json"; then
    echo "L=${L} manual compare: PASS"
  else
    rc=$?
    echo "L=${L} manual compare: FAIL (exit=${rc})"
    overall_pass=0
  fi
done

if [[ "${overall_pass}" == "1" ]]; then
  echo "REGRESSION PASS"
  exit 0
fi

echo "REGRESSION FAIL"
exit 1
