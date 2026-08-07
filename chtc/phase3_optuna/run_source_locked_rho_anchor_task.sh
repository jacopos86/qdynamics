#!/usr/bin/env bash
set -euo pipefail

CASE_ID="${1:?case id required}"
BATCH_ID="paper_i_source_locked_rho_anchors_20260609_v1"
SPEC_JSON="chtc/phase3_optuna/input/${BATCH_ID}/source_locked_rho_anchor_specs.json"
OUT_ROOT="raw_outputs/${BATCH_ID}"

mkdir -p "logs/${BATCH_ID}" "$OUT_ROOT"
python chtc/phase3_optuna/run_source_locked_rho_anchor.py \
  --case-id "$CASE_ID" \
  --spec-json "$SPEC_JSON" \
  --output-root "$OUT_ROOT"
