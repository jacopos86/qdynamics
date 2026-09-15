#!/usr/bin/env bash
# Inner runner for the Paper V CWRMF bounded-depth event-guard segments
# (runs inside the image).  One invocation is one bounded segment of the
# balanced event-triggered guard: it exits normally, leaves checkpoint.npz
# at the last accepted SSPRK2 step, and stages a receipt-named archive.
#
# Usage:
#   run_cwrmf_event_guard_accel.sh <run_id> <final_time> <time_step> \
#       <event_gram_hard_floor> <initial_state_file>
#
# <initial_state_file> is the previous segment's checkpoint.npz, relative to
# the job working directory; pass "none" for the first segment.
#
# Named options, all off unless the environment sets them.  The defaults are
# the recorded rule for the seed policy and this lane's declared solver
# setting set:
#   CWRMF_SOLVER_SETTING_PROFILE   default max_threads_8, which is this
#                                  lane's reason for requesting 8 CPUs; set
#                                  it to pinned_single_thread for the pinned
#                                  single-thread set
#   CWRMF_BUNDLE_SEED_POLICY       unset, i.e. monotone_hint.  Setting it to
#                                  carried_accepted_basis is a measured
#                                  regression on the balanced profile; read
#                                  the README section "Lane acceleration
#                                  decision" before using it
#   CWRMF_BUNDLE_SEED_RANK_MARGIN  read only when the policy is set
set -euo pipefail

RUN_ID="${1:?run_id required}"
FINAL_TIME="${2:?final_time required}"
TIME_STEP="${3:?time_step required}"
HARD_FLOOR="${4:?event_gram_hard_floor required}"
INITIAL_STATE_FILE="${5:-none}"

BATCH="paper_v_cwrmf_event_guard_accel_20260915_v1"
OUT_DIR="raw_outputs/${BATCH}/${RUN_ID}"
mkdir -p "raw_outputs/${BATCH}" "logs/${BATCH}" archives_by_receipt

export PYTHONPATH="paper_5/src${PYTHONPATH:+:${PYTHONPATH}}"

# The time-dynamics image carries numpy/scipy but not necessarily the conic
# stack.  Bootstrap a scratch venv with pinned versions only when an import
# is missing; never mutate the image or the submit-side environment.
if ! python3 -c "import numpy, scipy, cvxpy, clarabel" >/dev/null 2>&1; then
  echo "conic stack missing in image; bootstrapping scratch venv" >&2
  python3 -m venv --system-site-packages .paper5_venv
  # shellcheck disable=SC1091
  source .paper5_venv/bin/activate
  pip install --no-cache-dir --quiet \
    "numpy==2.3.5" "scipy==1.16.3" "cvxpy==1.9.2" "clarabel==0.11.1" \
    || pip install --no-cache-dir --quiet numpy scipy cvxpy clarabel
  python3 -c "import numpy, scipy, cvxpy, clarabel"
fi

ARGUMENTS=(
  --output-directory "${OUT_DIR}"
  --final-time "${FINAL_TIME}"
  --time-step "${TIME_STEP}"
  --numerical-profile balanced
  --event-triggered-guard
  --event-gram-hard-floor "${HARD_FLOOR}"
  --maximum-step-subdivisions 2
  --solver-setting-profile "${CWRMF_SOLVER_SETTING_PROFILE:-max_threads_8}"
  --tier-timing
  --compact-output
)

if [[ "${INITIAL_STATE_FILE}" != "none" && -n "${INITIAL_STATE_FILE}" ]]; then
  if [[ ! -f "${INITIAL_STATE_FILE}" ]]; then
    echo "declared initial state file is missing: ${INITIAL_STATE_FILE}" >&2
    exit 2
  fi
  ARGUMENTS+=(--initial-state-file "${INITIAL_STATE_FILE}")
fi

if [[ -n "${CWRMF_BUNDLE_SEED_POLICY:-}" ]]; then
  ARGUMENTS+=(--bundle-seed-policy "${CWRMF_BUNDLE_SEED_POLICY}")
  if [[ -n "${CWRMF_BUNDLE_SEED_RANK_MARGIN:-}" ]]; then
    ARGUMENTS+=(--bundle-seed-rank-margin "${CWRMF_BUNDLE_SEED_RANK_MARGIN}")
  fi
fi

echo "running: apcm_carried_witness_analysis ${ARGUMENTS[*]}" >&2
set +e
python3 -m paper5.stability.apcm_carried_witness_analysis "${ARGUMENTS[@]}"
DRIVER_STATUS=$?
set -e

# Stage the segment by receipt, never by name: the archive carries the
# campaign and the run in its file name, and it is placed in a directory
# named for its own SHA-256, so two stage-outs cannot overwrite each other.
ARCHIVE_NAME="${BATCH}__${RUN_ID}.tar.gz"
tar --exclude='__pycache__' -czf "${ARCHIVE_NAME}" "${OUT_DIR}"
if command -v sha256sum >/dev/null 2>&1; then
  ARCHIVE_SHA="$(sha256sum "${ARCHIVE_NAME}" | awk '{print $1}')"
else
  ARCHIVE_SHA="$(shasum -a 256 "${ARCHIVE_NAME}" | awk '{print $1}')"
fi
ARCHIVE_BYTES="$(wc -c < "${ARCHIVE_NAME}" | tr -d ' ')"
mkdir -p "archives_by_receipt/${ARCHIVE_SHA}"
mv "${ARCHIVE_NAME}" "archives_by_receipt/${ARCHIVE_SHA}/${ARCHIVE_NAME}"
cat > "archives_by_receipt/${ARCHIVE_SHA}/receipt.json" <<JSON
{
  "campaign": "${BATCH}",
  "run_id": "${RUN_ID}",
  "final_time": "${FINAL_TIME}",
  "time_step": "${TIME_STEP}",
  "event_gram_hard_floor": "${HARD_FLOOR}",
  "initial_state_file": "${INITIAL_STATE_FILE}",
  "solver_setting_profile": "${CWRMF_SOLVER_SETTING_PROFILE:-max_threads_8}",
  "bundle_seed_policy": "${CWRMF_BUNDLE_SEED_POLICY:-monotone_hint}",
  "archive_name": "${ARCHIVE_NAME}",
  "archive_sha256": "${ARCHIVE_SHA}",
  "archive_bytes": ${ARCHIVE_BYTES},
  "driver_exit_status": ${DRIVER_STATUS}
}
JSON
echo "archive_sha256=${ARCHIVE_SHA} archive_bytes=${ARCHIVE_BYTES}" >&2

exit "${DRIVER_STATUS}"
