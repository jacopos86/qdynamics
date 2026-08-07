#!/usr/bin/env bash
set -u -o pipefail

BATCH_ID="ap_mclachlan_a2_prune_diagnostic_20260704"
INPUT_DIR="chtc/${BATCH_ID}/input"
SEED_JSON="${INPUT_DIR}/a2_weakweak_append_no_guard_seed.json"
OUT_ROOT="raw_outputs/${BATCH_ID}"
LOG_ROOT="logs/${BATCH_ID}"
REF_DIR="${OUT_ROOT}/references"
FULL_DIR="${OUT_ROOT}/full"
SLIM_DIR="${OUT_ROOT}/slim"
STATUS_JSON="${OUT_ROOT}/job_status.json"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$REF_DIR" "$FULL_DIR" "$SLIM_DIR"

if [[ ! -f "$SEED_JSON" ]]; then
  echo "Missing seed artifact: $SEED_JSON" >&2
  exit 2
fi

KEEP_FULL="${APM_A2_DIAG_KEEP_FULL:-0}"
FAILED_STEPS=()
STARTED_AT="$(date -Iseconds)"

record_step_status() {
  local step="$1"
  local status="$2"
  local code="$3"
  python3 - "$STATUS_JSON" "$step" "$status" "$code" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
step, status, code = sys.argv[2], sys.argv[3], int(sys.argv[4])
payload = {}
if path.exists():
    payload = json.loads(path.read_text())
payload.setdefault("schema", "ap_mclachlan_a2_prune_diagnostic_status_v1")
payload.setdefault("started_at", None)
payload["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
steps = payload.setdefault("steps", {})
steps[step] = {"status": status, "exit_code": code}
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

initialize_status() {
  python3 - "$STATUS_JSON" "$STARTED_AT" "$SEED_JSON" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
started_at = sys.argv[2]
seed_path = Path(sys.argv[3])
payload = {
    "schema": "ap_mclachlan_a2_prune_diagnostic_status_v1",
    "diagnostic_only": True,
    "started_at": started_at,
    "seed_json": str(seed_path),
    "seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    "batch_id": "ap_mclachlan_a2_prune_diagnostic_20260704",
    "notes": [
        "Exact/reference trajectories are reporting-only.",
        "Drive-aligned density augmentation is not counted as tested append.",
        "Full trajectory JSONs are slimmed after each successful run to avoid transferring multi-GB payloads.",
    ],
    "steps": {},
}
status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

run_step() {
  local step="$1"
  shift
  echo "[$(date -Iseconds)] START ${step}" | tee -a "${LOG_ROOT}/job_progress.log"
  "$@" >"${LOG_ROOT}/${step}.out" 2>"${LOG_ROOT}/${step}.err"
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    echo "[$(date -Iseconds)] OK ${step}" | tee -a "${LOG_ROOT}/job_progress.log"
    record_step_status "$step" "ok" "$code"
  else
    echo "[$(date -Iseconds)] FAIL ${step} code=${code}" | tee -a "${LOG_ROOT}/job_progress.log"
    record_step_status "$step" "failed" "$code"
    FAILED_STEPS+=("${step}:${code}")
  fi
  return 0
}

slim_trajectory() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "Cannot slim missing trajectory: $src" >&2
    return 1
  fi
  python3 - "$src" "$dst" <<'PY'
import json
import math
import sys
from pathlib import Path
from typing import Any

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

SCALAR = (str, int, float, bool, type(None))
TOP_KEEP = {
    "schema",
    "decision_data_flow",
    "source_artifact_json",
    "drive_aligned_ansatz",
    "hamiltonian",
    "summary",
    "support_patch_summary",
}
DROP_ROW_KEYS = {
    "theta",
    "theta_runtime",
    "theta_dot",
    "state",
    "state_vector",
    "psi",
    "psi_t",
    "final_state",
    "initial_state",
    "candidate_scores",
    "patch_candidate_scores",
}


def scalar_or_short(value: Any) -> Any:
    if isinstance(value, SCALAR):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 24:
        out = []
        for item in value:
            if not isinstance(item, SCALAR):
                return None
            if isinstance(item, float) and not math.isfinite(item):
                out.append(None)
            else:
                out.append(item)
        return out
    return None


def slim_mapping(mapping: dict[str, Any], *, max_depth: int = 3) -> dict[str, Any]:
    def convert(value: Any, depth: int) -> Any:
        simple = scalar_or_short(value)
        if simple is not None or isinstance(value, type(None)):
            return simple
        if isinstance(value, dict) and depth < max_depth:
            out = {}
            for key, sub in value.items():
                converted = convert(sub, depth + 1)
                if converted is not None:
                    out[str(key)] = converted
            return out
        return None

    return {str(key): converted for key, value in mapping.items() if (converted := convert(value, 0)) is not None}


with src.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

slim: dict[str, Any] = {
    "report_slim_source": {
        "raw_trajectory_json": str(src),
        "slimmed_by": "chtc/ap_mclachlan_a2_prune_diagnostic_20260704/run_payload.sh",
    }
}
for key in TOP_KEEP:
    value = payload.get(key)
    if isinstance(value, dict):
        slim[key] = slim_mapping(value)
    elif value is not None:
        slim[key] = value

rows = payload.get("plot_rows")
if not isinstance(rows, list):
    rows = payload.get("trajectory")
if not isinstance(rows, list):
    rows = []
slim_rows = []
for row in rows:
    if not isinstance(row, dict):
        continue
    out = {}
    for key, value in row.items():
        if key in DROP_ROW_KEYS:
            continue
        simple = scalar_or_short(value)
        if simple is not None or value is None:
            out[str(key)] = simple
    slim_rows.append(out)
slim["plot_rows"] = slim_rows

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(slim, indent=2, sort_keys=True), encoding="utf-8")
PY
}

slim_after_success() {
  local step="$1"
  local src="$2"
  local dst="$3"
  if [[ -f "$src" ]]; then
    if slim_trajectory "$src" "$dst" >"${LOG_ROOT}/${step}_slim.out" 2>"${LOG_ROOT}/${step}_slim.err"; then
      record_step_status "${step}_slim" "ok" 0
      if [[ "$KEEP_FULL" != "1" ]]; then
        rm -f "$src"
      fi
    else
      local code=$?
      record_step_status "${step}_slim" "failed" "$code"
      FAILED_STEPS+=("${step}_slim:${code}")
    fi
  fi
}

COMMON_LOAD=(
  --artifact-json "$SEED_JSON"
  --loader-mode replay_family
  --generator-family match_adapt
  --fallback-family full_meta
  --parameterization-mode per_pauli_term
)

COMMON_TIME=(
  --t-final 3
  --num-times 601
)

COMMON_DRIVE=(
  --enable-drive
  --drive-A 0.6
  --drive-pattern staggered
  --drive-time-sampling midpoint
  --drive-t0 0.0
)

COMMON_SOLVE=(
  --integrator euler
  --pinv-rcond 1e-10
  --ridge-lambda 1e-7
  --solve-damping 0.0
)

APPEND_COMMON=(
  "${COMMON_LOAD[@]}"
  "${COMMON_TIME[@]}"
  "${COMMON_DRIVE[@]}"
  "${COMMON_SOLVE[@]}"
  --drive-aligned-ansatz
  --reference-energy-json "${REF_DIR}/exact_initial_ref.json"
  --seed-reference-energy-json "${REF_DIR}/seed_prepared_ref.json"
  --diagnostic-append-pool-mode replay_family_pool
  --require-complete-candidate-pool
  --append-ladder-mode combinatorial
  --max-append-batch-size 10
  --append-rung-set-cap 64
  --append-prefilter-size 12
  --append-min-time 0.005
  --residual-ratio-threshold 1e-3
  --append-cost-alpha 1.0
  --append-cost-lambda-2q 0.05
  --append-cost-lambda-d 0.05
  --append-cost-lambda-1q 0.025
  --append-cost-lambda-theta 0.0
  --append-cost-lambda-shot 0.02
  --solve-repair
  --solve-repair-condition-number-max 1.705e7
  --solve-repair-rho-num-max 1.0
  --solve-repair-state-motion-l2-step-max 5e-2
  --solve-repair-kink-eta-max 1e-2
  --solve-repair-local-subdivision
  --solve-repair-max-local-subdivisions 4
  --solve-repair-local-subdivision-factor 2
  --solve-repair-min-local-dt 1e-6
  --solve-repair-release-patience-min 1
  --solve-repair-release-patience-max 5
  --solve-repair-release-kink-threshold-scale 0.5
  --solve-repair-release-kink-severity-scale 4
  --solve-repair-ridge-ladder 1e-7,3e-8,1e-8,0,3e-7,1e-6,3e-6,1e-5
  --solve-repair-pinv-rcond-ladder 1e-10,1e-11,1e-12,1e-9,1e-8,1e-7
  --solve-repair-damping-ladder 0
  --progress-log-every 25
  --progress-log-events
)

PRUNE_COMMON=(
  --support-patch-prune
  --max-prune-batch-size 5
  --prune-rung-set-cap 64
  --prune-prefilter-size 12
  --prune-history-window 3
  --prune-history-lambda 3.0
  --prune-persistence-required 3
  --prune-persistence-mode atom_history
  --prune-atom-history-fraction 1.0
  --prune-condition-lambda-kappa-rel 0.02
  --prune-condition-lambda-schur 0.02
  --prune-condition-lambda-kappa-hist 0.04
  --prune-condition-lambda-kappa-dam 0.06
  --prune-ray-distance-tol 0.04
  --prune-differential-miss-tol 0.008
)

initialize_status

run_step pool_accounting \
  python3 -m pipelines.time_dynamics.diagnostics.ap_pool_accounting \
    "${COMMON_LOAD[@]}" \
    --paper-i-pool-json "$SEED_JSON" \
    --diagnostic-replay-family-pool \
    --output-json "${OUT_ROOT}/a2_pool_accounting.json"

run_step exact_reference \
  python3 -m pipelines.time_dynamics.runners.ap_reference_energy_from_adapt_artifact \
    "${COMMON_LOAD[@]}" \
    "${COMMON_TIME[@]}" \
    "${COMMON_DRIVE[@]}" \
    --reference-kind exact_initial_state_v1 \
    --method auto \
    --rtol 1e-10 \
    --atol 1e-12 \
    --output-json "${REF_DIR}/exact_initial_ref.json"

run_step seed_reference \
  python3 -m pipelines.time_dynamics.runners.ap_reference_energy_from_adapt_artifact \
    "${COMMON_LOAD[@]}" \
    "${COMMON_TIME[@]}" \
    "${COMMON_DRIVE[@]}" \
    --reference-kind seed_prepared_state_v1 \
    --method auto \
    --rtol 1e-10 \
    --atol 1e-12 \
    --output-json "${REF_DIR}/seed_prepared_ref.json"

run_step fixed_support \
  python3 -m pipelines.time_dynamics.runners.ap_fixed_from_adapt_artifact \
    "${COMMON_LOAD[@]}" \
    "${COMMON_TIME[@]}" \
    "${COMMON_DRIVE[@]}" \
    "${COMMON_SOLVE[@]}" \
    --drive-aligned-ansatz \
    --reference-energy-json "${REF_DIR}/exact_initial_ref.json" \
    --seed-reference-energy-json "${REF_DIR}/seed_prepared_ref.json" \
    --output-json "${FULL_DIR}/a2_fixed_no_tested_append_t3_n601.json"
slim_after_success fixed_support "${FULL_DIR}/a2_fixed_no_tested_append_t3_n601.json" "${SLIM_DIR}/a2_fixed_no_tested_append_t3_n601_report_slim.json"

run_step append_only \
  python3 -m pipelines.time_dynamics.runners.ap_append_from_adapt_artifact \
    "${APPEND_COMMON[@]}" \
    --output-json "${FULL_DIR}/a2_append_only_batchmax10_res1em3_t3_n601.json"
slim_after_success append_only "${FULL_DIR}/a2_append_only_batchmax10_res1em3_t3_n601.json" "${SLIM_DIR}/a2_append_only_batchmax10_res1em3_t3_n601_report_slim.json"

run_step prune_scoring \
  python3 -m pipelines.time_dynamics.runners.ap_append_from_adapt_artifact \
    "${APPEND_COMMON[@]}" \
    "${PRUNE_COMMON[@]}" \
    --output-json "${FULL_DIR}/a2_append_prune_score_batchmax10_atompersist3_t3_n601.json"
slim_after_success prune_scoring "${FULL_DIR}/a2_append_prune_score_batchmax10_atompersist3_t3_n601.json" "${SLIM_DIR}/a2_append_prune_score_batchmax10_atompersist3_t3_n601_report_slim.json"

run_step prune_commit \
  python3 -m pipelines.time_dynamics.runners.ap_append_from_adapt_artifact \
    "${APPEND_COMMON[@]}" \
    "${PRUNE_COMMON[@]}" \
    --support-patch-prune-commit \
    --no-prune-shadow-enabled \
    --max-prune-commits 10000 \
    --output-json "${FULL_DIR}/a2_append_prune_commit_batchmax10_atompersist3_shadowoff_t3_n601.json"
slim_after_success prune_commit "${FULL_DIR}/a2_append_prune_commit_batchmax10_atompersist3_shadowoff_t3_n601.json" "${SLIM_DIR}/a2_append_prune_commit_batchmax10_atompersist3_shadowoff_t3_n601_report_slim.json"

python3 - "$STATUS_JSON" "${#FAILED_STEPS[@]}" "${FAILED_STEPS[@]}" <<'PY'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

path = Path(sys.argv[1])
failure_count = int(sys.argv[2])
failures = list(sys.argv[3:])
payload = json.loads(path.read_text()) if path.exists() else {}
payload["finished_at"] = datetime.now(timezone.utc).isoformat()
payload["failure_count"] = failure_count
payload["failures"] = failures
payload["status"] = "ok" if failure_count == 0 else "partial_failed"
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY

if [[ "${#FAILED_STEPS[@]}" -gt 0 ]]; then
  printf 'Failed steps: %s\n' "${FAILED_STEPS[*]}" >&2
  exit 1
fi
exit 0
