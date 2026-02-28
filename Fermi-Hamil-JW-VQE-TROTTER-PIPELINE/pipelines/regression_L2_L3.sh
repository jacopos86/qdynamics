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
  --output-json "artifacts/json/H_L2_static_t${T_VAL}_U${U_VAL}_S${L2_TROTTER_STEPS}.json" \
  --output-pdf "artifacts/pdf/H_L2_static_t${T_VAL}_U${U_VAL}_S${L2_TROTTER_STEPS}.pdf" \
  --skip-pdf

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
  --output-json "artifacts/json/Q_L2_static_t${T_VAL}_U${U_VAL}_S${L2_TROTTER_STEPS}.json" \
  --output-pdf "artifacts/pdf/Q_L2_static_t${T_VAL}_U${U_VAL}_S${L2_TROTTER_STEPS}.pdf" \
  --skip-pdf

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
  --output-json "artifacts/json/H_L3_static_t${T_VAL}_U${U_VAL}_S${L3_TROTTER_STEPS}.json" \
  --output-pdf "artifacts/pdf/H_L3_static_t${T_VAL}_U${U_VAL}_S${L3_TROTTER_STEPS}.pdf" \
  --skip-pdf

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
  --output-json "artifacts/json/Q_L3_static_t${T_VAL}_U${U_VAL}_S${L3_TROTTER_STEPS}.json" \
  --output-pdf "artifacts/pdf/Q_L3_static_t${T_VAL}_U${U_VAL}_S${L3_TROTTER_STEPS}.pdf" \
  --skip-pdf

# No copy bridge needed: files are written directly to the paths
# that the compare pipeline expects under artifacts/json/.
# Separate compare calls per L because L=2 and L=3 use different trotter-steps,
# which affects the artifact tag.

echo "Running compare pipeline for L=2..."
"${PYTHON_BIN}" pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2 \
  --no-run-pipelines \
  --trotter-steps "${L2_TROTTER_STEPS}" \
  --t "${T_VAL}" --u "${U_VAL}" \
  --artifacts-dir artifacts \
  --skip-pdf

echo "Running compare pipeline for L=3..."
"${PYTHON_BIN}" pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 3 \
  --no-run-pipelines \
  --trotter-steps "${L3_TROTTER_STEPS}" \
  --t "${T_VAL}" --u "${U_VAL}" \
  --artifacts-dir artifacts \
  --skip-pdf

overall_pass=1
if [[ ! -f artifacts/json/HvQ_summary.json ]]; then
  overall_pass=0
else
  compare_all_pass="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/json/HvQ_summary.json")
data = json.loads(path.read_text(encoding="utf-8"))
print("1" if bool(data.get("all_pass", False)) else "0")
PY
)"
  if [[ "${compare_all_pass}" != "1" ]]; then
    overall_pass=0
  fi
fi

echo "Running manual compare checks..."
L2_TAG="L2_static_t${T_VAL}_U${U_VAL}_S${L2_TROTTER_STEPS}"
L3_TAG="L3_static_t${T_VAL}_U${U_VAL}_S${L3_TROTTER_STEPS}"
for tag in "${L2_TAG}" "${L3_TAG}"; do
  if "${PYTHON_BIN}" pipelines/manual_compare_jsons.py \
      --hardcoded "artifacts/json/H_${tag}.json" \
      --qiskit "artifacts/json/Q_${tag}.json" \
      --metrics "artifacts/json/HvQ_${tag}_metrics.json"; then
    echo "${tag} manual compare: PASS"
  else
    rc=$?
    echo "${tag} manual compare: FAIL (exit=${rc})"
    overall_pass=0
  fi
done

if [[ "${overall_pass}" == "1" ]]; then
  echo "REGRESSION PASS (static block)"
  # Fall through to drive block — overall_pass may still be set to 0 there.
else
  echo "REGRESSION FAIL (static block)"
  # Continue into drive block so we collect all failures before exiting.
fi

# ============================================================
# BLOCK 2: Drive-enabled regression (L=2 dimer_bias, L=3 staggered)
# ============================================================
#
# Drive parameters (fixed — change only with deliberate physical intent):
#   A=0.2, omega=1.7, tbar=4.0, phi=0.0
#   L=2: dimer_bias  |  L=3: staggered
#
# Short t_final=5.0 keeps piecewise-exact cheap.
# VQE is minimal; QPE is skipped (not relevant to drive physics).
#
# Physics sanity gate (applied to the hardcoded JSON only — independent
# of cross-pipeline comparison):
#   1. Final-time subspace fidelity (Trotter vs exact ground-manifold projector) >= 0.90
#   2. max(norm_before_renorm) over trajectory <= 1.0 + 5e-3
#
# Cross-pipeline compare uses relaxed subspace-fidelity threshold (5e-3) because
# drive introduces additional per-implementation divergence vs the static run.
# ============================================================

echo ""
echo "========================================================"
echo "BLOCK 2: Drive-enabled regression"
echo "========================================================"

DRIVE_ARTIFACTS_DIR="artifacts/drive_reg"
mkdir -p "${DRIVE_ARTIFACTS_DIR}/json"
mkdir -p "${DRIVE_ARTIFACTS_DIR}/pdf"

DRIVE_A="0.2"
DRIVE_OMEGA="1.7"
DRIVE_TBAR="4.0"
DRIVE_PHI="0.0"
DRIVE_T_FINAL="5.0"
DRIVE_NUM_TIMES="51"
DRIVE_TROTTER_STEPS="32"
DRIVE_VQE_REPS="1"
DRIVE_VQE_RESTARTS="1"
DRIVE_VQE_MAXITER="60"
DRIVE_TIME_SAMPLING="midpoint"

# Tag for drive artifacts (compare pipeline will generate the same).
DRIVE_TAG_L2="L2_vt_t${T_VAL}_U${U_VAL}_S${DRIVE_TROTTER_STEPS}"
DRIVE_TAG_L3="L3_vt_t${T_VAL}_U${U_VAL}_S${DRIVE_TROTTER_STEPS}"

# Physics sanity thresholds.
DRIVE_FIDELITY_FLOOR="0.90"
DRIVE_NORM_CEIL="1.005"

drive_pass=1

# ---- L=2 (dimer_bias) ----
echo "Running L=2 drive hardcoded pipeline (dimer_bias)..."
"${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
  --L 2 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final "${DRIVE_T_FINAL}" \
  --num-times "${DRIVE_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${DRIVE_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${DRIVE_VQE_REPS}" \
  --vqe-restarts "${DRIVE_VQE_RESTARTS}" \
  --vqe-maxiter "${DRIVE_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --skip-qpe \
  --initial-state-source exact \
  --enable-drive \
  --drive-A "${DRIVE_A}" \
  --drive-omega "${DRIVE_OMEGA}" \
  --drive-tbar "${DRIVE_TBAR}" \
  --drive-phi "${DRIVE_PHI}" \
  --drive-pattern dimer_bias \
  --drive-time-sampling "${DRIVE_TIME_SAMPLING}" \
  --output-json "${DRIVE_ARTIFACTS_DIR}/json/H_${DRIVE_TAG_L2}.json" \
  --output-pdf "${DRIVE_ARTIFACTS_DIR}/pdf/H_${DRIVE_TAG_L2}.pdf" \
  --skip-pdf

echo "Running L=2 drive qiskit pipeline (dimer_bias)..."
"${PYTHON_BIN}" pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 2 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final "${DRIVE_T_FINAL}" \
  --num-times "${DRIVE_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${DRIVE_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${DRIVE_VQE_REPS}" \
  --vqe-restarts "${DRIVE_VQE_RESTARTS}" \
  --vqe-maxiter "${DRIVE_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --skip-qpe \
  --initial-state-source exact \
  --enable-drive \
  --drive-A "${DRIVE_A}" \
  --drive-omega "${DRIVE_OMEGA}" \
  --drive-tbar "${DRIVE_TBAR}" \
  --drive-phi "${DRIVE_PHI}" \
  --drive-pattern dimer_bias \
  --drive-time-sampling "${DRIVE_TIME_SAMPLING}" \
  --output-json "${DRIVE_ARTIFACTS_DIR}/json/Q_${DRIVE_TAG_L2}.json" \
  --output-pdf "${DRIVE_ARTIFACTS_DIR}/pdf/Q_${DRIVE_TAG_L2}.pdf" \
  --skip-pdf

# ---- L=3 (staggered) ----
echo "Running L=3 drive hardcoded pipeline (staggered)..."
"${PYTHON_BIN}" pipelines/hardcoded_hubbard_pipeline.py \
  --L 3 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final "${DRIVE_T_FINAL}" \
  --num-times "${DRIVE_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${DRIVE_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${DRIVE_VQE_REPS}" \
  --vqe-restarts "${DRIVE_VQE_RESTARTS}" \
  --vqe-maxiter "${DRIVE_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --skip-qpe \
  --initial-state-source exact \
  --enable-drive \
  --drive-A "${DRIVE_A}" \
  --drive-omega "${DRIVE_OMEGA}" \
  --drive-tbar "${DRIVE_TBAR}" \
  --drive-phi "${DRIVE_PHI}" \
  --drive-pattern staggered \
  --drive-time-sampling "${DRIVE_TIME_SAMPLING}" \
  --output-json "${DRIVE_ARTIFACTS_DIR}/json/H_${DRIVE_TAG_L3}.json" \
  --output-pdf "${DRIVE_ARTIFACTS_DIR}/pdf/H_${DRIVE_TAG_L3}.pdf" \
  --skip-pdf

echo "Running L=3 drive qiskit pipeline (staggered)..."
"${PYTHON_BIN}" pipelines/qiskit_hubbard_baseline_pipeline.py \
  --L 3 \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --dv "${DV_VAL}" \
  --boundary "${BOUNDARY}" \
  --ordering "${ORDERING}" \
  --t-final "${DRIVE_T_FINAL}" \
  --num-times "${DRIVE_NUM_TIMES}" \
  --suzuki-order "${SUZUKI_ORDER}" \
  --trotter-steps "${DRIVE_TROTTER_STEPS}" \
  --term-order sorted \
  --vqe-reps "${DRIVE_VQE_REPS}" \
  --vqe-restarts "${DRIVE_VQE_RESTARTS}" \
  --vqe-maxiter "${DRIVE_VQE_MAXITER}" \
  --qpe-eval-qubits "${QPE_EVAL_QUBITS}" \
  --qpe-shots "${QPE_SHOTS}" \
  --qpe-seed "${QPE_SEED}" \
  --skip-qpe \
  --initial-state-source exact \
  --enable-drive \
  --drive-A "${DRIVE_A}" \
  --drive-omega "${DRIVE_OMEGA}" \
  --drive-tbar "${DRIVE_TBAR}" \
  --drive-phi "${DRIVE_PHI}" \
  --drive-pattern staggered \
  --drive-time-sampling "${DRIVE_TIME_SAMPLING}" \
  --output-json "${DRIVE_ARTIFACTS_DIR}/json/Q_${DRIVE_TAG_L3}.json" \
  --output-pdf "${DRIVE_ARTIFACTS_DIR}/pdf/Q_${DRIVE_TAG_L3}.pdf" \
  --skip-pdf

# ---- Cross-pipeline compare (drive artifacts dir, relaxed subspace-fidelity threshold) ----
echo "Running compare pipeline for drive L=2,3..."
"${PYTHON_BIN}" pipelines/compare_hardcoded_vs_qiskit_pipeline.py \
  --l-values 2,3 \
  --no-run-pipelines \
  --artifacts-dir "${DRIVE_ARTIFACTS_DIR}" \
  --t "${T_VAL}" \
  --u "${U_VAL}" \
  --trotter-steps "${DRIVE_TROTTER_STEPS}" \
  --enable-drive \
  --skip-pdf

drive_compare_pass="$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/drive_reg/json/HvQ_summary.json")
if not path.exists():
    print("0"); exit()
data = json.loads(path.read_text(encoding="utf-8"))
print("1" if bool(data.get("all_pass", False)) else "0")
PY
)"

if [[ "${drive_compare_pass}" != "1" ]]; then
  echo "DRIVE cross-pipeline compare: FAIL"
  drive_pass=0
else
  echo "DRIVE cross-pipeline compare: PASS"
fi

# ---- Manual JSON compare ----
echo "Running manual drive compare checks..."
for L in 2 3; do
  DTAG="L${L}_vt_t${T_VAL}_U${U_VAL}_S${DRIVE_TROTTER_STEPS}"
  if "${PYTHON_BIN}" pipelines/manual_compare_jsons.py \
      --hardcoded "${DRIVE_ARTIFACTS_DIR}/json/H_${DTAG}.json" \
      --qiskit "${DRIVE_ARTIFACTS_DIR}/json/Q_${DTAG}.json" \
      --metrics "${DRIVE_ARTIFACTS_DIR}/json/HvQ_${DTAG}_metrics.json"; then
    echo "DRIVE L=${L} manual compare: PASS"
  else
    rc=$?
    echo "DRIVE L=${L} manual compare: FAIL (exit=${rc})"
    drive_pass=0
  fi
done

# ---- Physics sanity gate ----
# Applied to the hardcoded pipeline JSON only (piecewise-exact reference is
# pipeline-internal, so this check is independent of Qiskit correctness).
echo "Running physics sanity gate..."
"${PYTHON_BIN}" - "${DRIVE_ARTIFACTS_DIR}" "${DRIVE_FIDELITY_FLOOR}" "${DRIVE_NORM_CEIL}" "${T_VAL}" "${U_VAL}" "${DRIVE_TROTTER_STEPS}" <<'PY'
import json
import sys
from pathlib import Path

artifacts_dir = Path(sys.argv[1])
fidelity_floor = float(sys.argv[2])
norm_ceil      = float(sys.argv[3])
t_val          = sys.argv[4]
u_val          = sys.argv[5]
steps          = sys.argv[6]

gate_pass = True
for L in (2, 3):
    tag  = f"L{L}_vt_t{t_val}_U{u_val}_S{steps}"
    path = artifacts_dir / "json" / f"H_{tag}.json"
    if not path.exists():
        print(f"  PHYSICS GATE L={L}: MISSING JSON -- FAIL")
        gate_pass = False
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    traj = data.get("trajectory", [])
    if not traj:
        print(f"  PHYSICS GATE L={L}: EMPTY TRAJECTORY -- FAIL")
        gate_pass = False
        continue

    # Gate 1: Final-time subspace fidelity.
    final_fidelity = float(traj[-1]["fidelity"])
    gate1 = final_fidelity >= fidelity_floor
    tag1  = "PASS" if gate1 else "FAIL"
    print(f"  PHYSICS GATE L={L} final_subspace_fidelity={final_fidelity:.6f} "
          f">= {fidelity_floor}: {tag1}")
    if not gate1:
        gate_pass = False

    # Gate 2: norm_before_renorm must stay within [0, norm_ceil] at all times.
    norms = [float(r["norm_before_renorm"]) for r in traj if "norm_before_renorm" in r]
    if not norms:
        print(f"  PHYSICS GATE L={L}: norm_before_renorm key absent -- SKIP")
    else:
        max_norm = max(norms)
        gate2 = max_norm <= norm_ceil
        tag2  = "PASS" if gate2 else "FAIL"
        print(f"  PHYSICS GATE L={L} max_norm_before_renorm={max_norm:.8f} "
              f"<= {norm_ceil}: {tag2}")
        if not gate2:
            gate_pass = False

    # Diagnostic: print subspace-fidelity range over trajectory (not a gate).
    fidelities = [float(r["fidelity"]) for r in traj]
    print(f"  DIAGNOSTIC  L={L} subspace_fidelity min={min(fidelities):.6f} "
          f"mean={sum(fidelities)/len(fidelities):.6f} final={fidelities[-1]:.6f}")

sys.exit(0 if gate_pass else 1)
PY
physics_gate_rc=$?
if [[ "${physics_gate_rc}" != "0" ]]; then
  echo "DRIVE physics sanity gate: FAIL"
  drive_pass=0
else
  echo "DRIVE physics sanity gate: PASS"
fi

# ---- Final verdict ----
echo ""
echo "========================================================"
if [[ "${overall_pass}" == "1" ]] && [[ "${drive_pass}" == "1" ]]; then
  echo "REGRESSION PASS"
  exit 0
fi

if [[ "${overall_pass}" != "1" ]]; then
  echo "REGRESSION FAIL (static block failed)"
fi
if [[ "${drive_pass}" != "1" ]]; then
  echo "REGRESSION FAIL (drive block failed)"
fi
exit 1
