#!/usr/bin/env bash
set -euo pipefail

# Preset runner for hardcoded + drive scaling studies.
# Goal: enforce VQE prep error gate
#   abs(vqe.energy - ground_state.exact_energy_filtered) < 1e-2
# while using per-L settings and deterministic fallbacks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required on PATH" >&2
  exit 2
fi

ERROR_THRESHOLD="${ERROR_THRESHOLD:-1e-2}"
L25_BUDGET_HOURS="${L25_BUDGET_HOURS:-10}"
RUN_L6="${RUN_L6:-1}"
SKIP_PDF="${SKIP_PDF:-1}"

L25_BUDGET_SEC="$(${PYTHON_BIN} - <<PY
print(int(float("${L25_BUDGET_HOURS}") * 3600.0))
PY
)"

# Fixed physics settings.
T_VAL="1.0"
U_VAL="4.0"
DV_VAL="0.0"
BOUNDARY="periodic"
ORDERING="blocked"
T_FINAL="20.0"
NUM_TIMES="201"
SUZUKI_ORDER="2"
TERM_ORDER="sorted"

# Fixed drive settings.
DRIVE_A="0.5"
DRIVE_OMEGA="2.0"
DRIVE_TBAR="3.0"
DRIVE_PHI="0.0"
DRIVE_PATTERN="staggered"
DRIVE_TIME_SAMPLING="midpoint"
DRIVE_T0="0.0"

VQE_SEED="7"

primary_params_for_L() {
  local L="$1"
  case "${L}" in
    2) echo "128 2 2 COBYLA 2 600" ;;
    3) echo "192 2 2 COBYLA 3 1200" ;;
    4) echo "256 3 3 SLSQP 2 80" ;;
    5) echo "384 4 3 SLSQP 1 90" ;;
    6) echo "512 4 3 SLSQP 1 120" ;;
    *)
      echo "ERROR: unsupported L=${L}" >&2
      return 1
      ;;
  esac
}

fallback1_params_for_L() {
  local L="$1"
  case "${L}" in
    2) echo "3 1200" ;;
    3) echo "4 2400" ;;
    4) echo "3 120" ;;
    5) echo "2 120" ;;
    6) echo "2 160" ;;
    *)
      echo "ERROR: unsupported L=${L}" >&2
      return 1
      ;;
  esac
}

fallback2_maxiter_for_L() {
  local L="$1"
  case "${L}" in
    4) echo "2600" ;;
    5) echo "3400" ;;
    6) echo "4200" ;;
    *)
      echo "ERROR: fallback2 unsupported for L=${L}" >&2
      return 1
      ;;
  esac
}

RUN_TS="$(date -u +"%Y%m%d_%H%M%S")"
ART_ROOT="artifacts/scaling_preset_L2_L6_${RUN_TS}"
JSON_DIR="${ART_ROOT}/json"
LOG_DIR="${ART_ROOT}/logs"
SUMMARY_TSV="${LOG_DIR}/summary.tsv"
BEST_TSV="${LOG_DIR}/best.tsv"

mkdir -p "${JSON_DIR}" "${LOG_DIR}"

echo -e "L\tattempt\tstatus\telapsed_sec\tvqe_error\tpass_threshold\tmethod\treps\trestarts\tmaxiter\ttrotter_steps\texact_steps_multiplier\tjson_path\tlog_path" > "${SUMMARY_TSV}"

RUN_ATTEMPT_STATUS=""
RUN_ATTEMPT_ERROR=""
RUN_ATTEMPT_JSON=""
RUN_ATTEMPT_LOG=""

is_pass_error_gate() {
  local err="$1"
  ${PYTHON_BIN} - "$err" "${ERROR_THRESHOLD}" <<'PY'
import math
import sys

try:
    e = float(sys.argv[1])
    thr = float(sys.argv[2])
except Exception:
    print("0")
    raise SystemExit(0)
print("1" if math.isfinite(e) and e < thr else "0")
PY
}

append_summary() {
  local L="$1"
  local attempt="$2"
  local status="$3"
  local elapsed="$4"
  local err="$5"
  local pass="$6"
  local method="$7"
  local reps="$8"
  local restarts="$9"
  local maxiter="${10}"
  local trotter="${11}"
  local exact_mult="${12}"
  local json_path="${13}"
  local log_path="${14}"

  echo -e "${L}\t${attempt}\t${status}\t${elapsed}\t${err}\t${pass}\t${method}\t${reps}\t${restarts}\t${maxiter}\t${trotter}\t${exact_mult}\t${json_path}\t${log_path}" >> "${SUMMARY_TSV}"
}

run_attempt() {
  local L="$1"
  local attempt="$2"
  local trotter="$3"
  local exact_mult="$4"
  local reps="$5"
  local method="$6"
  local restarts="$7"
  local maxiter="$8"

  local tag="${attempt}_L${L}_vt_t${T_VAL}_U${U_VAL}_S${trotter}"
  local json_path="${JSON_DIR}/${tag}.json"
  local log_path="${LOG_DIR}/${tag}.log"

  local start_s end_s elapsed cmd_rc
  start_s="$(date +%s)"
  cmd_rc=0

  local -a cmd=(
    "${PYTHON_BIN}" -u pipelines/hardcoded_hubbard_pipeline.py
    --L "${L}"
    --t "${T_VAL}"
    --u "${U_VAL}"
    --dv "${DV_VAL}"
    --boundary "${BOUNDARY}"
    --ordering "${ORDERING}"
    --t-final "${T_FINAL}"
    --num-times "${NUM_TIMES}"
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
    --skip-qpe
    --initial-state-source vqe
    --output-json "${json_path}"
  )

  if [[ "${SKIP_PDF}" == "1" ]]; then
    cmd+=(--skip-pdf)
  else
    cmd+=(--output-pdf "${ART_ROOT}/pdf/${tag}.pdf")
  fi

  {
    echo "# command_start $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf '%q ' "${cmd[@]}"
    echo
    "${cmd[@]}"
  } > "${log_path}" 2>&1 || cmd_rc=$?

  end_s="$(date +%s)"
  elapsed="$((end_s - start_s))"
  echo "# command_end rc=${cmd_rc} $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "${log_path}"

  local err="nan"
  if [[ -s "${json_path}" ]]; then
    err="$(jq -r '((.vqe.energy - .ground_state.exact_energy_filtered) | abs)' "${json_path}" 2>/dev/null || echo "nan")"
    [[ -n "${err}" ]] || err="nan"
    [[ "${err}" != "null" ]] || err="nan"
  fi

  local pass="0"
  if [[ "${cmd_rc}" -eq 0 ]]; then
    pass="$(is_pass_error_gate "${err}")"
  fi

  local status="FAIL_ERR_GATE"
  if [[ "${cmd_rc}" -ne 0 ]]; then
    status="CMD_FAIL"
    pass="0"
  elif [[ "${pass}" == "1" ]]; then
    status="PASS"
  fi

  append_summary "${L}" "${attempt}" "${status}" "${elapsed}" "${err}" "${pass}" \
    "${method}" "${reps}" "${restarts}" "${maxiter}" "${trotter}" "${exact_mult}" "${json_path}" "${log_path}"

  RUN_ATTEMPT_STATUS="${status}"
  RUN_ATTEMPT_ERROR="${err}"
  RUN_ATTEMPT_JSON="${json_path}"
  RUN_ATTEMPT_LOG="${log_path}"
}

run_with_fallback() {
  local L="$1"
  local p_trotter p_exact p_reps p_method p_restarts p_maxiter
  local fb1_restarts fb1_maxiter
  local fb2_maxiter

  read -r p_trotter p_exact p_reps p_method p_restarts p_maxiter < <(primary_params_for_L "${L}")

  run_attempt "${L}" "primary" \
    "${p_trotter}" "${p_exact}" "${p_reps}" \
    "${p_method}" "${p_restarts}" "${p_maxiter}"
  if [[ "${RUN_ATTEMPT_STATUS}" == "PASS" ]]; then
    return 0
  fi

  read -r fb1_restarts fb1_maxiter < <(fallback1_params_for_L "${L}")

  run_attempt "${L}" "fallback1" \
    "${p_trotter}" "${p_exact}" "${p_reps}" \
    "${p_method}" "${fb1_restarts}" "${fb1_maxiter}"
  if [[ "${RUN_ATTEMPT_STATUS}" == "PASS" ]]; then
    return 0
  fi

  if [[ "${L}" -ge 4 ]]; then
    fb2_maxiter="$(fallback2_maxiter_for_L "${L}")"
    run_attempt "${L}" "fallback2_cobyla" \
      "${p_trotter}" "${p_exact}" "3" \
      "COBYLA" "3" "${fb2_maxiter}"
    if [[ "${RUN_ATTEMPT_STATUS}" == "PASS" ]]; then
      return 0
    fi
  fi

  return 1
}

make_best_table() {
  ${PYTHON_BIN} - "${SUMMARY_TSV}" "${BEST_TSV}" <<'PY'
import csv
import math
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
best_path = pathlib.Path(sys.argv[2])

rows = []
with summary_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            L = int(row["L"])
        except Exception:
            continue
        row["_L"] = L
        try:
            row["_err"] = float(row["vqe_error"])
        except Exception:
            row["_err"] = float("nan")
        rows.append(row)

by_l = {}
for row in rows:
    by_l.setdefault(row["_L"], []).append(row)

with best_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["L", "status", "vqe_error", "attempt", "json_path"]) 

    for L in sorted(by_l):
        candidates = by_l[L]
        passing = [r for r in candidates if r.get("status") == "PASS"]

        if passing:
            chosen = min(
                passing,
                key=lambda r: r["_err"] if math.isfinite(r["_err"]) else float("inf"),
            )
        else:
            chosen = min(
                candidates,
                key=lambda r: (
                    0 if math.isfinite(r["_err"]) else 1,
                    r["_err"] if math.isfinite(r["_err"]) else float("inf"),
                ),
            )

        writer.writerow([
            str(L),
            chosen.get("status", ""),
            chosen.get("vqe_error", "nan"),
            chosen.get("attempt", ""),
            chosen.get("json_path", ""),
        ])
PY
}

overall_rc=0
l25_start="$(date +%s)"
budget_hit=0

for L in 2 3 4 5; do
  now="$(date +%s)"
  elapsed_l25="$((now - l25_start))"
  if [[ "${elapsed_l25}" -ge "${L25_BUDGET_SEC}" ]]; then
    budget_hit=1
    overall_rc=1
    for ((Lp=L; Lp<=5; Lp++)); do
      append_summary "${Lp}" "postponed_budget" "POSTPONED_BUDGET" "0" "nan" "0" "-" "-" "-" "-" "-" "-" "-" "-"
    done
    break
  fi

  if ! run_with_fallback "${L}"; then
    overall_rc=1
  fi
done

if [[ "${RUN_L6}" == "1" ]]; then
  if ! run_with_fallback 6; then
    overall_rc=1
  fi
else
  append_summary "6" "skipped" "SKIPPED_BY_CONFIG" "0" "nan" "0" "-" "-" "-" "-" "-" "-" "-" "-"
fi

make_best_table

{
  echo ""
  echo "=== Scaling Preset Run Complete ==="
  echo "Run root      : ${ART_ROOT}"
  echo "Summary table : ${SUMMARY_TSV}"
  echo "Best table    : ${BEST_TSV}"
  echo "Error gate    : ${ERROR_THRESHOLD}"
  echo "L2-L5 budget  : ${L25_BUDGET_HOURS}h"
  echo "Run L6        : ${RUN_L6}"
  if [[ "${budget_hit}" == "1" ]]; then
    echo "Budget status : HIT (L2-L5 remaining runs marked POSTPONED_BUDGET)"
  else
    echo "Budget status : not hit"
  fi
  echo ""
  echo "Best-by-L results:"
  cat "${BEST_TSV}"
} | tee "${LOG_DIR}/run_report.txt"

exit "${overall_rc}"
