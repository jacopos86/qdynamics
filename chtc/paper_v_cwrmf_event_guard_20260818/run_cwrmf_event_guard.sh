#!/usr/bin/env bash
# Inner runner for the Paper V CWRMF event-guard pilot (runs inside the image).
# Usage: run_cwrmf_event_guard.sh <run_id> <final_time> <time_step> <hard_floor>
set -euo pipefail

RUN_ID="${1:?run_id required}"
FINAL_TIME="${2:?final_time required}"
TIME_STEP="${3:?time_step required}"
HARD_FLOOR="${4:?event_gram_hard_floor required}"

BATCH="paper_v_cwrmf_event_guard_20260818_v1"
OUT_DIR="raw_outputs/${BATCH}/${RUN_ID}"
mkdir -p "raw_outputs/${BATCH}" "logs/${BATCH}"

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

python3 -m paper5.stability.apcm_carried_witness_analysis \
  --output-directory "${OUT_DIR}" \
  --final-time "${FINAL_TIME}" \
  --time-step "${TIME_STEP}" \
  --numerical-profile balanced \
  --event-triggered-guard \
  --event-gram-hard-floor "${HARD_FLOOR}" \
  --compact-output
