#!/usr/bin/env bash
set -euo pipefail

ROW_ID="${1:?row id required}"
BATCH_ID="paper_i_direct_command_rho_sweep_20260610_v1"
SPEC_JSON="chtc/phase3_optuna/input/${BATCH_ID}/direct_command_rho_sweep_specs.json"
OUT_ROOT="raw_outputs/${BATCH_ID}"

mkdir -p "logs/${BATCH_ID}" "$OUT_ROOT"
python chtc/phase3_optuna/run_direct_command_rho_sweep.py \
  --row-id "$ROW_ID" \
  --spec-json "$SPEC_JSON" \
  --output-root "$OUT_ROOT"
