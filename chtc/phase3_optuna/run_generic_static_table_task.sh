#!/usr/bin/env bash
set -euo pipefail

RECORD_ID="${1:?record_id required}"
RECORDS_PATH="${2:-${GENERIC_STATIC_TABLE_RECORDS_PATH:-chtc/phase3_optuna/input/generic_static_table_records.tsv}}"
OUT_ROOT="${3:-raw_outputs/generic_static_table/${RECORD_ID}}"

mkdir -p logs "$(dirname "$OUT_ROOT")"
python -u -m chtc.phase3_optuna.generic_static_table_runner "$RECORD_ID" "$RECORDS_PATH" "$OUT_ROOT"
