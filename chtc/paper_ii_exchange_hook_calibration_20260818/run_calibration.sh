#!/usr/bin/env bash
set -euo pipefail
CONFIG_ID="${1:?config_id required}"
BASE=chtc/paper_ii_exchange_hook_calibration_20260818
OUT_ROOT="raw_outputs/paper_ii_exchange_hook_calibration_20260818/${CONFIG_ID}"
mkdir -p "$OUT_ROOT" logs
EXTRA=$(awk -F'\t' -v id="$CONFIG_ID" '$1==id {print $2}' "$BASE/input/configs.tsv")
if [[ -z "$EXTRA" ]]; then
  echo "config_id ${CONFIG_ID} not found" >&2
  exit 3
fi
REFIT_FLAGS=(--certification-refit --certification-refit-trust-radius 0.6 --certification-refit-max-iterations 40)
if [[ "$EXTRA" == *"--strip-refit"* ]]; then
  REFIT_FLAGS=()
  EXTRA="${EXTRA//--strip-refit/}"
fi
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
# shellcheck disable=SC2086
python3 pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py \
  --artifact-json "$BASE/input/seed_intweak_nph3.json" \
  --output-json "$OUT_ROOT/run.json" \
  --t-final "${CALIB_T_FINAL:-1.0}" --num-times "${CALIB_NUM_TIMES:-26}" \
  --residual-ratio-threshold 0.0 \
  --max-joint-patch-evaluations 50000 \
  --max-certification-attempts-per-level 6 \
  --max-structural-pool-size 4 \
  --prune-target-policy appended_only \
  --progress-log-every 1 \
  "${REFIT_FLAGS[@]}" \
  $EXTRA
