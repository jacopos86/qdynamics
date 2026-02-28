#!/usr/bin/env bash
set -euo pipefail

# Shorthand runner:
#   "run L=<n>" => drive-enabled (never static), accuracy-gated.
#
# Gate:
#   abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-7
#
# Strategy:
#   1) Deterministic per-L primary settings
#   2) Fallback A (more optimizer effort)
#   3) Fallback B (deeper ansatz + finer dynamics controls)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Usage:
  bash pipelines/run_L_drive_accurate.sh --L <int> [options]

Options:
  --L <int>                Lattice size (required)
  --artifacts-dir <path>   Output root (default: artifacts)
  --budget-hours <float>   Attempt budget in hours (default: 6 for L<=4, else 10)
  --python-bin <path>      Python executable (default: /opt/anaconda3/bin/python or python)
  --skip-pdf               Skip PDF generation (default)
  --with-pdf               Generate PDF
  -h, --help               Show this help
USAGE
}

L_VAL=""
ARTIFACTS_DIR="artifacts"
BUDGET_HOURS=""
SKIP_PDF="1"
PYTHON_BIN_DEFAULT="/opt/anaconda3/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --L)
      L_VAL="${2:-}"
      shift 2
      ;;
    --artifacts-dir)
      ARTIFACTS_DIR="${2:-}"
      shift 2
      ;;
    --budget-hours)
      BUDGET_HOURS="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --skip-pdf)
      SKIP_PDF="1"
      shift
      ;;
    --with-pdf)
      SKIP_PDF="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${L_VAL}" ]]; then
  echo "ERROR: --L is required" >&2
  usage
  exit 2
fi

if ! [[ "${L_VAL}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --L must be an integer, got '${L_VAL}'" >&2
  exit 2
fi
if (( L_VAL < 2 )); then
  echo "ERROR: --L must be >= 2, got '${L_VAL}'" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: python executable not found: ${PYTHON_BIN}" >&2
    exit 2
  fi
fi

if [[ -z "${BUDGET_HOURS}" ]]; then
  if (( L_VAL <= 4 )); then
    BUDGET_HOURS="6"
  else
    BUDGET_HOURS="10"
  fi
fi

ERROR_THRESHOLD="1e-7"

# Fixed model defaults
T_VAL="1.0"
U_VAL="4.0"
DV_VAL="0.0"
BOUNDARY="periodic"
ORDERING="blocked"
SUZUKI_ORDER="2"
TERM_ORDER="sorted"
FIDELITY_TOL="1e-8"
VQE_SEED="7"

# Fixed shorthand drive profile (always enabled)
DRIVE_A="0.5"
DRIVE_OMEGA="2.0"
DRIVE_TBAR="3.0"
DRIVE_PHI="0.0"
DRIVE_PATTERN="staggered"
DRIVE_TIME_SAMPLING="midpoint"
DRIVE_T0="0.0"

round_to_64() {
  local n="$1"
  echo $(( ((n + 32) / 64) * 64 ))
}

primary_params_for_L() {
  local L="$1"
  case "${L}" in
    2) echo "128 2 201 2 2 COBYLA 1200" ;;
    3) echo "192 2 201 2 3 COBYLA 2400" ;;
    4) echo "256 3 241 4 4 SLSQP 6000" ;;
    5) echo "384 3 301 4 5 SLSQP 8000" ;;
    6) echo "512 4 361 5 6 SLSQP 10000" ;;
    *)
      "${PYTHON_BIN}" - "${L}" <<'PY'
import math
import sys
L = int(sys.argv[1])
trotter = int(round((128 * (L - 2)) / 64.0) * 64)
exact_mult = min(6, 2 + ((L - 1) // 2))
reps = min(6, 3 + ((L - 3) // 2))
restarts = min(8, 4 + ((L - 3) // 2))
maxiter = 6000 + 1500 * (L - 4)
num_times = min(401, 201 + 40 * (L - 2))
method = "SLSQP"
print(f"{trotter} {exact_mult} {num_times} {reps} {restarts} {method} {maxiter}")
PY
      ;;
  esac
}

delta_e_from_json() {
  local json_path="$1"
  "${PYTHON_BIN}" - "${json_path}" <<'PY'
import json
import math
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("nan")
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding="utf-8"))
    v = float(data["vqe"]["energy"])
    e = float(data["ground_state"]["exact_energy_filtered"])
    d = abs(v - e)
    print(d if math.isfinite(d) else "nan")
except Exception:
    print("nan")
PY
}

is_gate_pass() {
  local err="$1"
  "${PYTHON_BIN}" - "${err}" "${ERROR_THRESHOLD}" <<'PY'
import math
import sys
try:
    err = float(sys.argv[1])
    thr = float(sys.argv[2])
except Exception:
    print("0")
    raise SystemExit(0)
print("1" if math.isfinite(err) and err < thr else "0")
PY
}

BUDGET_SEC="$("${PYTHON_BIN}" - "${BUDGET_HOURS}" <<'PY'
import sys
print(int(float(sys.argv[1]) * 3600.0))
PY
)"

RUN_TS="$(date -u +"%Y%m%d_%H%M%S")"
RUN_ROOT="${ARTIFACTS_DIR}/run_L_drive_accurate_L${L_VAL}_${RUN_TS}"
JSON_DIR="${RUN_ROOT}/json"
LOG_DIR="${RUN_ROOT}/logs"
PDF_DIR="${RUN_ROOT}/pdf"
SUMMARY_TSV="${LOG_DIR}/attempts.tsv"
SUMMARY_TXT="${LOG_DIR}/summary.txt"

mkdir -p "${JSON_DIR}" "${LOG_DIR}"
if [[ "${SKIP_PDF}" == "0" ]]; then
  mkdir -p "${PDF_DIR}"
fi

echo -e "attempt\tstatus\telapsed_sec\tdelta_e\tthreshold\tpassed\ttrotter_steps\texact_steps_multiplier\tnum_times\tvqe_reps\tvqe_restarts\tvqe_method\tvqe_maxiter\tjson_path\tlog_path" > "${SUMMARY_TSV}"

RUN_START_SEC="$(date +%s)"
BEST_DELTA="nan"
BEST_JSON=""
BEST_ATTEMPT=""
FINAL_STATUS="FAIL_GATE"

record_attempt() {
  local attempt="$1"
  local status="$2"
  local elapsed="$3"
  local delta="$4"
  local passed="$5"
  local trotter="$6"
  local exact_mult="$7"
  local num_times="$8"
  local reps="$9"
  local restarts="${10}"
  local method="${11}"
  local maxiter="${12}"
  local json_path="${13}"
  local log_path="${14}"
  echo -e "${attempt}\t${status}\t${elapsed}\t${delta}\t${ERROR_THRESHOLD}\t${passed}\t${trotter}\t${exact_mult}\t${num_times}\t${reps}\t${restarts}\t${method}\t${maxiter}\t${json_path}\t${log_path}" >> "${SUMMARY_TSV}"
}

maybe_update_best() {
  local attempt="$1"
  local delta="$2"
  local json_path="$3"
  local is_better
  is_better="$("${PYTHON_BIN}" - "${BEST_DELTA}" "${delta}" <<'PY'
import math
import sys
try:
    cur = float(sys.argv[1])
except Exception:
    cur = float("nan")
try:
    new = float(sys.argv[2])
except Exception:
    new = float("nan")
if not math.isfinite(new):
    print("0")
elif not math.isfinite(cur):
    print("1")
else:
    print("1" if new < cur else "0")
PY
)"
  if [[ "${is_better}" == "1" ]]; then
    BEST_DELTA="${delta}"
    BEST_JSON="${json_path}"
    BEST_ATTEMPT="${attempt}"
  fi
}

budget_ok() {
  local now
  now="$(date +%s)"
  (( now - RUN_START_SEC < BUDGET_SEC ))
}

run_attempt() {
  local attempt="$1"
  local trotter="$2"
  local exact_mult="$3"
  local num_times="$4"
  local reps="$5"
  local restarts="$6"
  local method="$7"
  local maxiter="$8"

  local tag="${attempt}_L${L_VAL}_vt_t${T_VAL}_U${U_VAL}_S${trotter}"
  local json_path="${JSON_DIR}/${tag}.json"
  local log_path="${LOG_DIR}/${tag}.log"
  local pdf_path="${PDF_DIR}/${tag}.pdf"

  local start_s end_s elapsed rc
  start_s="$(date +%s)"
  rc=0

  local -a cmd=(
    "${PYTHON_BIN}" -u pipelines/hardcoded_hubbard_pipeline.py
    --L "${L_VAL}"
    --t "${T_VAL}"
    --u "${U_VAL}"
    --dv "${DV_VAL}"
    --boundary "${BOUNDARY}"
    --ordering "${ORDERING}"
    --t-final 20.0
    --num-times "${num_times}"
    --suzuki-order "${SUZUKI_ORDER}"
    --trotter-steps "${trotter}"
    --term-order "${TERM_ORDER}"
    --enable-drive
    --drive-A "${DRIVE_A}"
    --drive-omega "${DRIVE_OMEGA}"
    --drive-tbar "${DRIVE_TBAR}"
    --drive-phi "${DRIVE_PHI}"
    --drive-pattern "${DRIVE_PATTERN}"
    --drive-time-sampling "${DRIVE_TIME_SAMPLING}"
    --drive-t0 "${DRIVE_T0}"
    --exact-steps-multiplier "${exact_mult}"
    --vqe-reps "${reps}"
    --vqe-restarts "${restarts}"
    --vqe-seed "${VQE_SEED}"
    --vqe-maxiter "${maxiter}"
    --vqe-method "${method}"
    --fidelity-subspace-energy-tol "${FIDELITY_TOL}"
    --skip-qpe
    --initial-state-source vqe
    --output-json "${json_path}"
  )

  if [[ "${SKIP_PDF}" == "1" ]]; then
    cmd+=(--skip-pdf)
  else
    cmd+=(--output-pdf "${pdf_path}")
  fi

  {
    echo "# command_start $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
  } > "${log_path}" 2>&1 || rc=$?

  end_s="$(date +%s)"
  elapsed="$((end_s - start_s))"
  echo "# command_end rc=${rc} $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${log_path}"

  local delta="nan"
  local passed="0"
  local status="CMD_FAIL"
  if [[ "${rc}" -eq 0 ]]; then
    delta="$(delta_e_from_json "${json_path}")"
    passed="$(is_gate_pass "${delta}")"
    if [[ "${passed}" == "1" ]]; then
      status="PASS"
    else
      status="FAIL_GATE"
    fi
  fi

  record_attempt "${attempt}" "${status}" "${elapsed}" "${delta}" "${passed}" \
    "${trotter}" "${exact_mult}" "${num_times}" "${reps}" "${restarts}" "${method}" "${maxiter}" \
    "${json_path}" "${log_path}"
  maybe_update_best "${attempt}" "${delta}" "${json_path}"

  if [[ "${status}" == "PASS" ]]; then
    FINAL_STATUS="PASS"
    return 0
  fi
  return 1
}

read -r p_trotter p_exact p_num_times p_reps p_restarts p_method p_maxiter < <(primary_params_for_L "${L_VAL}")

if budget_ok; then
  run_attempt "primary" "${p_trotter}" "${p_exact}" "${p_num_times}" "${p_reps}" "${p_restarts}" "${p_method}" "${p_maxiter}" || true
else
  FINAL_STATUS="BUDGET_EXHAUSTED"
fi

if [[ "${FINAL_STATUS}" != "PASS" && "${FINAL_STATUS}" != "BUDGET_EXHAUSTED" ]]; then
  if budget_ok; then
    fb_a_restarts="$((p_restarts + 2))"
    fb_a_maxiter="$((p_maxiter * 2))"
    fb_a_method="${p_method}"
    if (( L_VAL >= 4 )); then
      fb_a_method="L-BFGS-B"
    fi
    run_attempt "fallback_A" "${p_trotter}" "${p_exact}" "${p_num_times}" "${p_reps}" "${fb_a_restarts}" "${fb_a_method}" "${fb_a_maxiter}" || true
  else
    FINAL_STATUS="BUDGET_EXHAUSTED"
  fi
fi

if [[ "${FINAL_STATUS}" != "PASS" && "${FINAL_STATUS}" != "BUDGET_EXHAUSTED" ]]; then
  if budget_ok; then
    fb_b_trotter="$(round_to_64 $(( (p_trotter * 3) / 2 )) )"
    fb_b_exact="$((p_exact + 1))"
    fb_b_reps="$((p_reps + 1))"
    fb_b_restarts="$((p_restarts + 2))"
    fb_b_maxiter="$((p_maxiter * 2))"
    fb_b_method="${p_method}"
    if (( L_VAL >= 4 )); then
      fb_b_method="L-BFGS-B"
    fi
    run_attempt "fallback_B" "${fb_b_trotter}" "${fb_b_exact}" "${p_num_times}" "${fb_b_reps}" "${fb_b_restarts}" "${fb_b_method}" "${fb_b_maxiter}" || true
  else
    FINAL_STATUS="BUDGET_EXHAUSTED"
  fi
fi

if [[ "${FINAL_STATUS}" != "PASS" && "${FINAL_STATUS}" != "BUDGET_EXHAUSTED" ]]; then
  FINAL_STATUS="FAIL_GATE"
fi

{
  echo "Run summary (drive-only accurate shorthand)"
  echo "L: ${L_VAL}"
  echo "status: ${FINAL_STATUS}"
  echo "threshold_delta_e: ${ERROR_THRESHOLD}"
  echo "best_delta_e: ${BEST_DELTA}"
  echo "best_attempt: ${BEST_ATTEMPT}"
  echo "best_json: ${BEST_JSON}"
  echo "budget_hours: ${BUDGET_HOURS}"
  echo "artifacts_root: ${RUN_ROOT}"
  echo "attempt_table: ${SUMMARY_TSV}"
} | tee "${SUMMARY_TXT}"

if [[ "${FINAL_STATUS}" == "PASS" ]]; then
  exit 0
fi
exit 1
