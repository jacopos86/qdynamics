#!/usr/bin/env bash
set -euo pipefail
BASE=chtc/paper_iii_regime_handoff_20260819
OUT_ROOT="raw_outputs/paper_iii_regime_handoff_20260819"
mkdir -p "$OUT_ROOT" logs
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
python3 pipelines/excited_dynamics/paper_iii_regime_handoff.py \
  --source-seed-json "$BASE/input/source_seed.json" \
  --omega "${HANDOFF_OMEGA:-1.25}" \
  --t-final "${HANDOFF_T_FINAL:-8.0}" \
  --num-steps "${HANDOFF_NUM_STEPS:-160}" \
  --dts "${HANDOFF_DTS:-0.05}" \
  --output-dir "$OUT_ROOT"
