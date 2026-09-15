#!/usr/bin/env bash
# Inner runner for the Paper V CWRMF bounded-depth event-guard segments
# (runs inside the image).  One invocation is one bounded segment of the
# balanced event-triggered guard: it exits normally, leaves checkpoint.npz
# at the last accepted SSPRK2 step, and stages a receipt-named archive.
#
# Usage:
#   run_cwrmf_event_guard_accel.sh <run_id> <final_time> <time_step> \
#       <event_gram_hard_floor> <initial_state_file> [<expected_initial_time>]
#
# <initial_state_file> is the checkpoint.npz the segment resumes from,
# relative to the job working directory; pass "none" for the first segment.
# For the next segment it is the previous segment's; for a resume of a
# segment stopped at its wall budget it is that segment's own.
#
# <expected_initial_time> is the time the declared checkpoint is supposed to
# hold, or "none" to declare nothing.  It is passed to the driver as
# --initial-time, which refuses a checkpoint whose recorded time disagrees.
# That is what turns a stale checkpoint path -- the previous segment's
# stopped checkpoint instead of its resumed one -- from a silent start at
# the wrong time into a refusal.
#
# <run_id> is used once.  The runner refuses a run id that already owns a
# directory under raw_outputs/<batch>/ (exit 3), and
# make_resume_submit.py in this directory writes every continuation with a
# fresh one, so no submit file has to be hand-edited to resume.
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
#   CWRMF_MAXIMUM_WALL_SECONDS     default 138240, which is 0.8 x the
#                                  +MaxRuntime = 172800 s the submit files
#                                  declare.  The rollout stops at the first
#                                  accepted step past it, writes its
#                                  checkpoint and exits 0, so the segment
#                                  ends normally and stages out instead of
#                                  being held at the pool's 72 h wall with
#                                  nothing transferred.  The remaining fifth
#                                  of the runtime is the headroom the
#                                  offline exact-reference scoring, the
#                                  artifact writing and the archiving need.
set -euo pipefail

RUN_ID="${1:?run_id required}"
FINAL_TIME="${2:?final_time required}"
TIME_STEP="${3:?time_step required}"
HARD_FLOOR="${4:?event_gram_hard_floor required}"
INITIAL_STATE_FILE="${5:-none}"
EXPECTED_INITIAL_TIME="${6:-none}"

BATCH="paper_v_cwrmf_event_guard_accel_20260915_v1"
MAXIMUM_WALL_SECONDS="${CWRMF_MAXIMUM_WALL_SECONDS:-138240}"
OUT_DIR="raw_outputs/${BATCH}/${RUN_ID}"

# A run id is used once.  The destructive case this refuses is a submission
# that resumes a segment into the run id that wrote the checkpoint: the
# checkpoint arrives in the sandbox at raw_outputs/<batch>/<run_id>/, this
# job's own output directory, and the stage-out would then replace on the
# access point the very checkpoint the segment was resumed from with a
# shorter one.  Refusing here costs nothing -- it is before the conic
# bootstrap and before the driver -- and the exit status is its own.
# chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py names
# every continuation with a fresh run id, so a generated submit file cannot
# reach this branch.
if [[ -e "${OUT_DIR}" ]]; then
  echo "run id ${RUN_ID} already owns ${OUT_DIR} in this sandbox;" >&2
  echo "a run id is used once -- generate the continuation with" >&2
  echo "  chtc/paper_v_cwrmf_event_guard_accel_20260915/make_resume_submit.py" >&2
  exit 3
fi
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
  --maximum-wall-seconds "${MAXIMUM_WALL_SECONDS}"
  --tier-timing
  --compact-output
)

if [[ "${INITIAL_STATE_FILE}" != "none" && -n "${INITIAL_STATE_FILE}" ]]; then
  if [[ ! -f "${INITIAL_STATE_FILE}" ]]; then
    echo "declared initial state file is missing: ${INITIAL_STATE_FILE}" >&2
    exit 2
  fi
  ARGUMENTS+=(--initial-state-file "${INITIAL_STATE_FILE}")
  if [[ "${EXPECTED_INITIAL_TIME}" != "none" && -n "${EXPECTED_INITIAL_TIME}" ]]; then
    # The driver compares this against the checkpoint's own recorded time and
    # refuses the pair when they disagree, so a checkpoint path left over
    # from a superseded run cannot start this segment at the wrong time.
    ARGUMENTS+=(--initial-time "${EXPECTED_INITIAL_TIME}")
  fi
elif [[ "${EXPECTED_INITIAL_TIME}" != "none" && -n "${EXPECTED_INITIAL_TIME}" \
        && "${EXPECTED_INITIAL_TIME}" != "0" && "${EXPECTED_INITIAL_TIME}" != "0.0" ]]; then
  # Declaring a nonzero start with no checkpoint would start the prepared
  # ground state at that time, which is a different physical run, not a
  # continuation.
  echo "expected_initial_time ${EXPECTED_INITIAL_TIME} was declared with no" >&2
  echo "initial_state_file; a nonzero start needs the checkpoint that holds it" >&2
  exit 2
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

# What the segment reached, read from the run's own summary.json.  A segment
# stopped at the declared wall budget exits 0 exactly like one that reached
# final_time, so stop_reason and last_time are the only things that separate
# them; both go in the receipt, because the next submission is chosen from
# the receipt.  stop_reason wall_budget means this segment's own final_time
# must be submitted again from the checkpoint it left.
SUMMARY_PATH="${OUT_DIR}/summary.json"
SEGMENT_STATE="$(python3 -c 'import json, sys
block = json.load(open(sys.argv[1], encoding="utf-8"))["integration"]
print(block.get("stop_reason", "unrecorded"),
      block.get("last_time", "unrecorded"),
      block.get("initial_time", "unrecorded"))' \
  "${SUMMARY_PATH}" 2>/dev/null || true)"
if [[ -z "${SEGMENT_STATE}" ]]; then
  SEGMENT_STATE="unrecorded unrecorded unrecorded"
fi
read -r STOP_REASON LAST_TIME INITIAL_TIME <<< "${SEGMENT_STATE}"

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
  "expected_initial_time": "${EXPECTED_INITIAL_TIME}",
  "maximum_wall_seconds": "${MAXIMUM_WALL_SECONDS}",
  "stop_reason": "${STOP_REASON}",
  "last_time": "${LAST_TIME}",
  "initial_time": "${INITIAL_TIME}",
  "solver_setting_profile": "${CWRMF_SOLVER_SETTING_PROFILE:-max_threads_8}",
  "bundle_seed_policy": "${CWRMF_BUNDLE_SEED_POLICY:-monotone_hint}",
  "bundle_seed_rank_margin": "${CWRMF_BUNDLE_SEED_RANK_MARGIN:-none}",
  "archive_name": "${ARCHIVE_NAME}",
  "archive_sha256": "${ARCHIVE_SHA}",
  "archive_bytes": ${ARCHIVE_BYTES},
  "driver_exit_status": ${DRIVER_STATUS}
}
JSON
echo "archive_sha256=${ARCHIVE_SHA} archive_bytes=${ARCHIVE_BYTES}" >&2
echo "stop_reason=${STOP_REASON} last_time=${LAST_TIME}" >&2

exit "${DRIVER_STATUS}"
